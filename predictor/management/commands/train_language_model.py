from django.core.management.base import BaseCommand

from predictor.ml import MODEL_PATH, get_model_metrics, train_and_save_model


class Command(BaseCommand):
    help = "Train the language prediction model, save the .pkl artifact, and print accuracy."

    def handle(self, *args, **options):
        train_and_save_model()
        metrics = get_model_metrics()
        self.stdout.write(self.style.SUCCESS(f"Model saved to: {MODEL_PATH}"))
        self.stdout.write(self.style.SUCCESS(f"Accuracy: {metrics['accuracy_percent']}%"))
