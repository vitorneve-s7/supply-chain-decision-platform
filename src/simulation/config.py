"""Settings of the supply chain network: one object describes one scenario."""

from dataclasses import dataclass


@dataclass
class NetworkConfig:
    """Lead times, capacities and one optional disruption.

    DC -> store lead times are not here: they come from draw_lead_times (Phase 2), per item and
    order day, so the twin can be checked against the Phase 2 replay.
    """

    supplier_lead: tuple[int, int] = (5, 9)  # supplier -> DC, uniform whole days
    # Capacities: units shipped in any 7 days, x average weekly store demand (None = unlimited)
    dc_capacity: float | None = 1.5
    supplier_capacity: float | None = 1.5
    dc_start_days: float = 14  # starting DC stock, in days of each item's average demand
    unlimited_dc: bool = False  # True: the DC always has stock and capacity (Phase 2 check)

    # One disruption, active on replay days start_day .. start_day + duration - 1
    start_day: int | None = None
    duration: int = 0
    supplier_down: bool = False  # S1: the supplier ships nothing
    demand_multiplier: float = 1.0  # S2: store demand x this
    dc_capacity_multiplier: float = 1.0  # S3: DC capacity x this
    store_lead_multiplier: float = 1.0  # S4: DC -> store lead time x this

    def disrupted(self, day: int) -> bool:
        """True if the disruption is active on this replay day."""
        if self.start_day is None:
            return False
        return self.start_day <= day < self.start_day + self.duration
