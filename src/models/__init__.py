from enum import Enum


class NetType(Enum):
    cdp = 'Concrete droppath',
    mc = 'Monte-Carlo dropout',
    mc_df = 'Monte-Carlo dropfilter',
    mc_dp = 'Monte-Carlo droppath',
    sdp = 'Scheduled droppath'
    vanilla = 'vanilla',
