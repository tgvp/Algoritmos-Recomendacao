import numpy as np
import pandas as pd
from tqdm import tqdm

from similarity import calculate_similarity, calculate_similarity_matrix, calculate_similarity_gpu

class Recommmendation:
    def __init__(self, user_item_matrix: pd.DataFrame, target_user_id: int, top_n: int = 3):
        """Recommendation constructor

        Args:
            user_item_matrix (pd.DataFrame): The user x item matrix as pandas.DataFrame (including index and columns)
            target_user_id (int): targetUser ID
            top_n (int, optional): Number of items that will be recommended. Defaults to top 3 items.
        """
        
        self.top_n = top_n
        self.target_user_id = target_user_id
        self.user_item_matrix = user_item_matrix
        
        self.similarity_matrix = calculate_similarity_matrix(user_item_matrix,
                                                             verbose=False)
        
        if 'cluster' in self.user_item_matrix.columns:
            pass
        self.target_user_cluster = int(self.user_item_matrix.loc[self.target_user_id]['cluster'])
        self.users_same_cluster = self.user_item_matrix[self.user_item_matrix['cluster'] == self.target_user_cluster].index
        self.users_same_cluster = self.users_same_cluster[self.users_same_cluster != self.target_user_id]
        self.target_user_ratings = self.user_item_matrix.loc[self.target_user_id].drop('cluster')
                
    def get_matrix(self) -> pd.DataFrame:
        """Returns the user-item matrix

        Returns:
            pd.DataFrame: User-item matrix
        """
        return self.user_item_matrix
    
    @classmethod
    def get_unrated_items(cls):
        # filter out items that the target user has been already rated
        cls.unrated_items = cls.target_user_ratings[cls.target_user_ratings <= 0].index
        
    
    @classmethod
    def get_similarity_matrix(cls) -> pd.DataFrame:
        """Returns the similarity matrix

        Returns:
            pd.DataFrame: Similarity matrix
        """
        return cls.similarity_matrix

    @classmethod
    def get_user_rated_items(cls) -> list[str]:
        """Returns the list of items rated by the given user.

        Args:
            user_id (str): User ID
            user_item_matrix (pd.DataFrame): User-item matrix

        Returns:
            list[str]: List of items rated by the given user
        """
        return list(cls.user_item_matrix.loc[cls.target_user_id][cls.user_item_matrix.loc[cls.target_user_id] > 0].index)[:-1]
    
    
    def predict_rating(target_user_ratings: pd.Series, neighbor_user_ratings: pd.Series, user_similarity: float, item: str) -> float:
        """Predicts the rating of an item for a target user in comparison to a neighbor user which has rated the item and is in the same cluster.

        Args:
            target_user_ratings (pd.Series): Ratings of the target user
            neighbor_user_ratings (pd.Series): Ratings of the neighbor user
            user_similarity (float): Similarity between the target user and the neighbor user
            item (str): Item name

        Returns:
            float: Predicted rating
        """
        
        numerator = np.sum(user_similarity * (neighbor_user_ratings[item] - np.mean(neighbor_user_ratings)))
        denominator = np.sum(np.abs(user_similarity))
        
        if denominator == 0:
            denominator = 1e-6
        
        return np.mean(target_user_ratings) + numerator / denominator


    def collaborative_filtering_within_clusters(self, verbose=False) -> list[str]:
        """Performs collaborative filtering within clusters. Where the target user is in a cluster and the recommendations are made from the same cluster.

        Args:
            verbose (bool, optional): Print intermediate results. Defaults to False.

        Returns:
            list[str]: List of recommended items
        
        """
        
        # initializing recommendations with zeros
        recommendations = pd.Series(0, index=self.target_user_ratings.index, dtype=float)
        #print(recommendations)
        
        for user in tqdm.tqdm(self.users_same_cluster, desc='Processing recommendations for users in the same cluster: '):
            
            user_ratings = self.user_item_matrix.loc[user].drop('cluster')
            
            user_similarity = calculate_similarity(target_user_ratings, user_ratings, similarity)
            
            
            # predict ratings for unrated items
            for item in self.unrated_items:
                
                # predict the rating using collaborative filtering
                predicted_rating = predict_rating(target_user_ratings, user_ratings, user_similarity, item)
                
                recommendations[item] += predicted_rating

        # sorting
        sorted_recommendations = recommendations.sort_values(ascending=False).index
        
        # get top-n recommendations
        top_recommendations = sorted_recommendations[:num_recommendations]
        
        if verbose:
            print(f'Target user: {target_user}')
            print(f'Target user cluster: {target_user_cluster}\n')
            print(f'Users in the same cluster: {users_same_cluster}\n')
            print(f'Target user ratings:\n{target_user_ratings}\n')
            print(f'Sorted recommendations:\n{sorted_recommendations}\n')
            print(f'Top recommendations:\n{top_recommendations}')
        
        return list(top_recommendations)
    
    def __repr__(self) -> str:
        return f'Recommendation(target_user_id={self.target_user_id}, top_n={self.top_n})'
    
    def __str__(self) -> str:
        return f'Recommendation(target_user_id={self.target_user_id}, top_n={self.top_n})'