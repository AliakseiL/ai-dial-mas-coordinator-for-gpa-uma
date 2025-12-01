import json
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage
from pydantic import StrictStr

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        # 1. Create AsyncDial client (api_version='2025-01-01-preview')
        # 2. Open stage for Coordination Request (StageProcessor will help with that)
        # 3. Prepare coordination request
        # 4. Add to the stage generated coordination request and close the stage
        # 5. Handle coordination request (don't forget that all the content that will write called agent need to provide to stage)
        # 6. Generate final response based on the message from called agent
        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version='2025-01-01-preview'
        )
        coordination_stage = StageProcessor.open_stage(choice=choice, name="Coordination Request")
        coordination_request = await self.__prepare_coordination_request(client, request)
        logger.info(f"coordination_request: {coordination_request.model_dump_json()}")
        coordination_stage.append_content(f"```json\n\r{coordination_request.model_dump_json(indent=2)}\n\r```\n\r")
        StageProcessor.close_stage_safely(coordination_stage)

        process_stage = StageProcessor.open_stage(choice=choice, name="Coordinated Process")
        agent_message = await self.__handle_coordination_request(
            coordination_request=coordination_request,
            choice=choice,
            stage=process_stage,
            request=request
        )
        StageProcessor.close_stage_safely(process_stage)

        logger.info(f"agent_message: {agent_message.json()}")

        final_message = await self.__final_response(
            client=client,
            choice=choice,
            request=request,
            agent_message=agent_message
        )
        logger.info(f"final_message: {final_message.json()}")

        return final_message


    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequest:
        # 1. Make call to LLM with prepared messages and COORDINATION_REQUEST_SYSTEM_PROMPT. For GPT model we can use
        #    `response_format` https://platform.openai.com/docs/guides/structured-outputs?example=structured-data and
        #    response will be returned in JSON format. The `response_format` parameter must be provided as extra_body dict
        #    {response_format": {"type": "json_schema","json_schema": {"name": "response","schema": CoordinationRequest.model_json_schema()}}}
        # 2. Get content from response -> choice -> message -> content
        # 3. Load as dict
        # 4. Create CoordinationRequest from result, since CoordinationRequest is pydentic model, you can use `model_validate` method

        messages = self.__prepare_messages(request, COORDINATION_REQUEST_SYSTEM_PROMPT)
        response = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=messages,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": CoordinationRequest.model_json_schema()
                    }
                }
            }
        )
        response_content = response.choices[0].message.content
        response_dict = json.loads(response_content)
        coordination_request = CoordinationRequest.model_validate(response_dict)
        return coordination_request

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        # 1. Create array with messages, first message is system prompt and it is dict
        # 2. Iterate through messages from request and:
        #       - if user message that it has custom content and then add dict with user message and content (custom_content should be skipped)
        #       - otherwise append it as dict with excluded none fields (use `dict` method, despite it is deprecated since
        #         DIAL is using pydentic.v1)
        messages: list[dict[str, Any]] = []
        system_message = {"role": Role.SYSTEM, "content": system_prompt}
        messages.append(system_message)
        for msg in request.messages:
            if msg.role == Role.USER and msg.custom_content is not None:
                user_message = {"role": Role.USER, "content": msg.content}
                messages.append(user_message)
            else:
                messages.append(msg.dict(exclude_none=True))
        return messages

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request
    ) -> Message:
        # Make appropriate coordination requests to proper agents and return the result
        if coordination_request.agent_name == AgentName.GPA:
            gateway = GPAGateway(endpoint=self.endpoint)
        else:
            gateway = UMSAgentGateway(ums_agent_endpoint=self.ums_agent_endpoint)

        agent_message = await gateway.response(choice, stage, request, coordination_request.additional_instructions)
        return agent_message


    async def __final_response(
            self, client: AsyncDial,
            choice: Choice,
            request: Request,
            agent_message: Message
    ) -> Message:
        # 1. Prepare messages with FINAL_RESPONSE_SYSTEM_PROMPT
        # 2. Make augmentation of retrieved agent response (as context) with user request (as user request)
        # 3. Update last message content with augmented prompt
        # 4. Call LLM with streaming
        # 5. Stream final response to choice
        messages = self.__prepare_messages(request, FINAL_RESPONSE_SYSTEM_PROMPT)
        # Augmentation
        augmented_user_message = deepcopy(messages[-1])
        augmented_user_message['content'] += f"\n\nContext from the agent:\n{agent_message.content}"
        messages[-1] = augmented_user_message

        final_response_chunks = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=messages,
            stream=True
        )

        final_message_content = ""

        async for chunk in final_response_chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content'):
                    content_delta = delta.content
                    if content_delta:
                        final_message_content += content_delta
                        choice.append_content(content_delta)

        return Message(role=Role.ASSISTANT, content=StrictStr(final_message_content), custom_content=agent_message.custom_content)
