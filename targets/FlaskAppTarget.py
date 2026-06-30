import httpx
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import PromptChatTarget


class FlaskAppTarget(PromptChatTarget):
    def __init__(self):
        super().__init__()
        self.endpoint = "http://localhost:5002/api/chat"

    def _extract_text(self, msg) -> str:
        if hasattr(msg, "content"):
            return msg.content
        elif hasattr(msg, "text"):
            return msg.text
        elif hasattr(msg, "value"):
            return msg.value
        else:
            try:
                msg_dict = msg.dict() if hasattr(msg, "dict") else vars(msg)
                return msg_dict.get("text", msg_dict.get("content", str(msg)))
            except Exception:
                return str(msg)

    async def _send_prompt_to_target_async(
        self, *, normalized_conversation: list
    ) -> Message:
        history_lines = []
        for piece_msg in normalized_conversation[:-1]:
            role = getattr(piece_msg, "role", "user")
            text = self._extract_text(piece_msg)
            history_lines.append(f"{role}: {text}")

        latest_msg = normalized_conversation[-1]
        prompt_text = self._extract_text(latest_msg)

        if history_lines:
            full_payload = "\n".join(history_lines) + f"\nuser: {prompt_text}"
        else:
            full_payload = prompt_text

        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(self.endpoint, json={"message": full_payload})
            data = response.json()
            answer = data.get("response", "No response")
            return [
                Message(
                    message_pieces=[
                        MessagePiece(
                            role="assistant",
                            original_value=answer,
                            converted_value=answer,
                            original_value_data_type="text",
                            converted_value_data_type="text",
                        )
                    ]
                )
            ]
