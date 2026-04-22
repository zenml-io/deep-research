"""Base model with strict configuration for all V2 contracts."""

from pydantic import BaseModel, ConfigDict


class StrictBase(BaseModel):
    """Base model that rejects unknown fields.

    All V2 data contracts inherit from this to ensure structural
    integrity — no silent field drops, no surprise extras.
    """

    model_config = ConfigDict(extra="forbid")
