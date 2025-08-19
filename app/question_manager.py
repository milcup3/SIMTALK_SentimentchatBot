import pandas as pd
import random
import json
from typing import List, Dict, Any

class QuestionManager:
    def __init__(self, sct_path: str):
        self.sct_df = pd.read_json(sct_path, lines=True)

    def get_categories(self) -> List[str]:
        return self.sct_df['category'].unique().tolist()

    def select_category(self, idx: int) -> str:
        cats = self.get_categories()
        return cats[idx] if 0 <= idx < len(cats) else cats[0]

    def get_questions_by_category(self, category: str) -> List[str]:
        return self.sct_df[self.sct_df['category'] == category]['conversational_template'].tolist()

    def get_random_question(self, category: str) -> str:
        pool = self.get_questions_by_category(category)
        return random.choice(pool) if pool else "그 부분에 대해 이야기해 주세요."

    def get_all(self) -> List[Dict[str, Any]]:
        return self.sct_df.to_dict('records')
