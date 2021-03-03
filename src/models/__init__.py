from enum import Enum


class NetType(Enum):
    vanilla = 'vanilla',
    mc = 'Monte-Carlo dropout',
    mc_dp = 'Monte-Carlo droppath',
    mc_df = 'Monte-Carlo dropfilter',
    sdp = 'Scheduled droppath'
