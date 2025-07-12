# /auction_sim/auction.py
import numpy as np
from typing import Dict, List
from . import config

class GSPAuction:
    """
    实现广义第二价格拍卖 (Generalized Second-Price Auction, GSP)
    """
    def __init__(self, n_slots: int, ctr_positions: np.ndarray, ctr_noise_std: float):
        self.n_slots = n_slots
        
        # 确保CTR_POSITIONS数组长度与N_SLOTS匹配
        if len(ctr_positions) != n_slots:
            if len(ctr_positions) < n_slots:
                # 如果CTR数组太短，用递减值填充
                print(f"Warning: CTR_POSITIONS length ({len(ctr_positions)}) < N_SLOTS ({n_slots})")
                additional_ctrs = []
                last_ctr = ctr_positions[-1]
                step = last_ctr / (n_slots - len(ctr_positions) + 1)
                for i in range(n_slots - len(ctr_positions)):
                    last_ctr -= step
                    additional_ctrs.append(max(0.1, last_ctr))  # 最小CTR为0.1
                self.ctr_positions = np.concatenate([ctr_positions, additional_ctrs])
            else:
                # 如果CTR数组太长，截取前n_slots个
                print(f"Warning: CTR_POSITIONS length ({len(ctr_positions)}) > N_SLOTS ({n_slots}), truncating")
                self.ctr_positions = ctr_positions[:n_slots]
        else:
            self.ctr_positions = ctr_positions
            
        self.ctr_noise_std = ctr_noise_std
        print(f"GSP Auction initialized with {n_slots} slots, CTR: {self.ctr_positions}")

    def run_auction(self, bids: Dict[str, float]) -> Dict[str, Dict]:
        """
        运行一轮GSP拍卖。

        Args:
            bids (Dict[str, float]): 一个字典，key是agent_id，value是出价。
                                     {'agent_A': 1.2, 'agent_B': 1.5}

        Returns:
            Dict[str, Dict]: 一个结果字典，key是agent_id，value包含其排名、支付成本和赢得的槽位CTR。
                             {
                                'agent_B': {'rank': 1, 'cost_per_click': 1.2, 'slot_ctr': 0.7...},
                                'agent_A': {'rank': 2, 'cost_per_click': 0, 'slot_ctr': 0.3...},
                                'agent_C': {'rank': 3, 'cost_per_click': 0, 'slot_ctr': 0},
                             }
        """
        if not bids:
            return {}

        # 1. 对出价进行降序排序
        sorted_bidders = sorted(bids.items(), key=lambda item: item[1], reverse=True)
        
        # 2. 确定赢家和支付价格
        winners = sorted_bidders[:self.n_slots]
        losers = sorted_bidders[self.n_slots:]
        
        results = {}
        
        # 为本轮拍卖生成带噪声的CTR
        noisy_ctrs = self.ctr_positions * (1 + np.random.uniform(-self.ctr_noise_std, self.ctr_noise_std, size=self.n_slots))
        noisy_ctrs = np.clip(noisy_ctrs, 0, 1) # 确保CTR在[0,1]范围内

        # 3. 计算赢家的成本和信息
        for i in range(len(winners)):
            agent_id, bid_price = winners[i]
            
            # 支付价格是下一位的出价
            if i + 1 < len(winners):
                cost_per_click = winners[i+1][1]
            elif losers: # 如果是最后一个赢家，支付第一名输家的出价
                cost_per_click = losers[0][1]
            else: # 如果赢家数量小于广告位数且没有输家
                cost_per_click = 0.0

            results[agent_id] = {
                'rank': i + 1,
                'won': True,
                'bid': bid_price,
                'cost_per_click': cost_per_click,
                'slot_ctr': noisy_ctrs[i]
            }
            
        # 4. 记录输家的信息
        for i in range(len(losers)):
            agent_id, bid_price = losers[i]
            results[agent_id] = {
                'rank': len(winners) + i + 1,
                'won': False,
                'bid': bid_price,
                'cost_per_click': 0.0,
                'slot_ctr': 0.0
            }
            
        return results