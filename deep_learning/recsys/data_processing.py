"""


"""
from torch.utils.data import Dataset
import logging
logger = logging.getLogger()
RANDOM_STATE = 11


class CapitalConnectData(Dataset):
    """ .   industry_code (b2b_dataset_management  ) -> BERT -> ndarry (768) -> 
    company_hq_state 50+ states -> [] 4-8 dimension 
    ? text features 在什么时候转化为tensor?
    
    Parameters:
    -----------
    
    Returns:
    --------
        [item_features, user_features], target
    """

    def __init__(self, df, cfg) -> None:
        super().__init__()

        self.data = df
        self.percent = cfg.percent
        if self.percent >= 1:
            self.data = self.data[:self.percent]
        else:
            self.data = self.data[:int(self.percent*len(self.data))]
        
        self.model_type = cfg.model_type

        self.deal_features = cfg.deal_features
        self.deal_features_dense = cfg.deal_features_dense
        self.deal_features_sparse = cfg.deal_features_sparse
        self.deal_features_num = cfg.deal_features_num
        self.deal_features_text = cfg.deal_features_text
        
        self.inv_features = cfg.inv_features
        self.inv_features_dense = cfg.inv_features_dense
        self.inv_features_sparse = cfg.inv_features_sparse
        self.inv_features_num = cfg.inv_features_num
        self.inv_features_text = cfg.inv_features_text

        if self.model_type == "two_tower":
            self.target == cfg.target
        


