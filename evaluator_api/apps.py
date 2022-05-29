from django.apps import AppConfig

from evaluator_api.utils.load_eval_model import load_img_eval_model
from evaluator_api.utils.load_eval_model import load_text_eval_model


class EvaluatorApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'evaluator_api'

    def ready(self):
        """
        Loading Evaluation Models
        """

        print("Initiating Loading Model Sequence...")
        print("Loading IMAGE_CLF model...")
        load_img_eval_model()
        print("Loading IMAGE_CLF - Complete.")
        print("Loading TEXT_CLF model...")
        load_text_eval_model()
        print("Loading TEXT_CLF - Complete.")
