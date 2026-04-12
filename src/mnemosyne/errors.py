import uuid


class MemoryNotFound(Exception):
    def __init__(self, memory_id: uuid.UUID):
        self.memory_id = memory_id
        super().__init__(f"Memory {memory_id} not found")


class CannedResponseMissing(Exception):
    def __init__(self, text: str):
        super().__init__(f"No canned response for input: {text[:80]!r}")


class RuleLoadError(Exception):
    pass


class MalformedLLMResponse(Exception):
    pass
