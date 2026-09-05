import re
from enum import Enum
from typing import Dict, Any, Tuple

class AllowedRemediationAction(str, Enum):
    ACTION_RESTART_CONTAINER = "ACTION_RESTART_CONTAINER"
    ACTION_FLUSH_REDIS_CACHE = "ACTION_FLUSH_REDIS_CACHE"
    ACTION_SCALE_SERVICE = "ACTION_SCALE_SERVICE"
    ACTION_ROLLBACK = "ACTION_ROLLBACK"
    ACTION_NO_OP = "ACTION_NO_OP"

class GuardrailViolationError(Exception):
    """Raised when an LLM remediation action fails strict enum or parameter validation."""
    pass

class ActionGuardrailController:
    ALLOWED_ACTIONS = {action.value for action in AllowedRemediationAction}
    SAFE_TARGET_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{2,64}$")

    ACTION_RISK_MAP = {
        AllowedRemediationAction.ACTION_RESTART_CONTAINER: "LOW_RISK",
        AllowedRemediationAction.ACTION_FLUSH_REDIS_CACHE: "LOW_RISK",
        AllowedRemediationAction.ACTION_SCALE_SERVICE: "HIGH_BLAST_RADIUS",
        AllowedRemediationAction.ACTION_ROLLBACK: "HIGH_BLAST_RADIUS",
        AllowedRemediationAction.ACTION_NO_OP: "LOW_RISK",
    }

    @classmethod
    def get_risk_level(cls, remediation_key: str) -> str:
        return cls.ACTION_RISK_MAP.get(remediation_key, "HIGH_BLAST_RADIUS")

    @classmethod
    def validate_action(cls, remediation_key: str, target_name: str) -> Tuple[bool, str]:
        """
        Validates remediation key against strict Enum allow-list and checks target name sanitization.
        Returns (is_valid, validation_message).
        """
        # 1. Reject any non-string or unknown action key
        if not isinstance(remediation_key, str) or remediation_key not in cls.ALLOWED_ACTIONS:
            msg = f"GUARDRAIL REJECTION: Remediation key '{remediation_key}' is not in strict allow-list {cls.ALLOWED_ACTIONS}."
            raise GuardrailViolationError(msg)

        # 2. Sanitize and validate target container/service name
        if not isinstance(target_name, str) or not cls.SAFE_TARGET_REGEX.match(target_name):
            msg = f"GUARDRAIL REJECTION: Target name '{target_name}' contains illegal characters or invalid format."
            raise GuardrailViolationError(msg)

        # 3. Prevent command injection attempts
        forbidden_substrings = [";", "&&", "||", "`", "$", "(", ")", ">", "<", "|", "../", "/etc/", "rm "]
        for forbidden in forbidden_substrings:
            if forbidden in remediation_key or forbidden in target_name:
                msg = f"GUARDRAIL REJECTION: Injection attempt detected containing '{forbidden}'."
                raise GuardrailViolationError(msg)

        return True, f"GUARDRAIL PASSED: Action '{remediation_key}' on target '{target_name}' is verified safe."

