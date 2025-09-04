import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm


class BinaryClassifierTrainer:
    def __init__(self, model, criterion, optimizer, device, dataloader):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.dataloader = dataloader
        self.optimizer = optimizer

    def _step(self, batch, running_loss, correct, total):
        x_batch = batch["pixel_values"].to(self.device, torch.float32)
        y_batch = batch["labels"].to(self.device).unsqueeze(1).float()
        outputs = self.model(x_batch)
        loss = self.criterion(outputs, y_batch)

        running_loss += loss.item() * x_batch.size(0)
        logit = torch.sigmoid(outputs)
        predicted = (logit >= 0.5).long()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

        return running_loss, correct, total, loss, predicted, logit

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.dataloader["train"]):
            self.optimizer.zero_grad()
            running_loss, correct, total, loss, _, _ = self._step(batch, running_loss, correct, total)
            loss.backward()
            self.optimizer.step()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, split="val", init_test=False):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_pred, val_gt, val_logit = [], [], []
        with torch.no_grad():
            for batch in tqdm(self.dataloader[split]):
                running_loss, correct, total, _, predicted, logit = self._step(batch, running_loss, correct, total)
                val_gt.append(batch["labels"].float())
                val_pred.append(predicted.squeeze(-1).cpu())
                val_logit.append(logit.squeeze(-1).cpu())
                if init_test:
                    break

        val_pred = torch.cat(val_pred).numpy()
        val_gt = torch.cat(val_gt).numpy()
        val_logit = torch.cat(val_logit).numpy()

        val_loss = running_loss / total
        val_acc = correct / total

        f1 = f1_score(val_gt, val_pred)
        precision = precision_score(val_gt, val_pred)
        recall = recall_score(val_gt, val_pred)

        metrics = {
            "loss": val_loss,
            "accuracy": val_acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        return val_gt, val_pred, val_logit, metrics

    def fit(self, num_epochs, save_dir):
        best_val_loss = float("inf")
        with torch.no_grad():
            val_gt, val_pred, _, metrics = self.evaluate(init_test=True)
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_gt, val_pred, _, metrics = self.evaluate()

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {metrics['loss']:.4f}, Val Acc: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}"
            )

            if metrics["loss"] < best_val_loss:
                best_val_loss = metrics["loss"]
                torch.save(self.model.state_dict(), f"{save_dir}/best_model.pth")
                print(f"✅ Model saved at epoch {epoch+1} with val_loss {metrics['loss']:.4f}")

        weights = torch.load(f"{save_dir}/best_model.pth", map_location=self.device)
        self.model.load_state_dict(weights)
        with torch.no_grad():
            eval_gt, eval_pred, eval_logit, eval_metrics = self.evaluate(split="test")

        print("==== Test Results ====")
        print(
            f"Test Loss: {eval_metrics['loss']:.4f}, Test Acc: {eval_metrics['accuracy']:.4f}, "
            f"Test F1: {eval_metrics['f1']:.4f}, Test Precision: {eval_metrics['precision']:.4f}, Test Recall: {eval_metrics['recall']:.4f}"
        )

        df_test_preds = pd.DataFrame({"predictions": eval_pred}, {"targets": eval_gt}, {"logits": eval_logit})
        df_test_preds.to_csv(f"{save_dir}/test_predictions.csv", index=False)


class MultiClassClassifierTrainer(BinaryClassifierTrainer):
    def _step(self, batch, running_loss, correct, total):
        x_batch = batch["pixel_values"].to(self.device, torch.float32)
        y_batch = batch["labels"].to(self.device).long()  # Ensure labels are of type long

        outputs = self.model(x_batch)
        loss = self.criterion(outputs, y_batch)

        running_loss += loss.item() * x_batch.size(0)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

        return running_loss, correct, total, loss, predicted, outputs

    @torch.no_grad()
    def evaluate(self, split="val", init_test=False):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_pred, val_gt, val_logit = [], [], []
        with torch.no_grad():
            for batch in tqdm(self.dataloader[split]):
                running_loss, correct, total, loss, predicted, logits = self._step(batch, running_loss, correct, total)
                val_gt.append(batch["labels"].cpu())
                val_pred.append(predicted.cpu())
                val_logit.append(logits.cpu())
                if init_test:
                    break

        val_pred = torch.cat(val_pred).numpy()
        val_gt = torch.cat(val_gt).numpy()
        val_logit = torch.cat(val_logit).softmax(dim=1).numpy()

        val_loss = running_loss / total
        val_acc = correct / total

        f1 = f1_score(val_gt, val_pred, average="weighted")
        precision = precision_score(val_gt, val_pred, average="weighted")
        recall = recall_score(val_gt, val_pred, average="weighted")

        metrics = {
            "loss": val_loss,
            "accuracy": val_acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        return val_gt, val_pred, val_logit, metrics

    def fit(self, num_epochs, save_dir):
        best_acc = 0
        with torch.no_grad():
            val_gt, val_pred, _, metrics = self.evaluate(init_test=True)
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_gt, val_pred, val_logit, metrics = self.evaluate()

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {metrics['loss']:.4f}, Val Acc: {metrics['accuracy']:.4f}, "
            )

            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                torch.save(self.model.state_dict(), f"{save_dir}/best_model.pth")
                print(f"✅ Model saved at epoch {epoch+1} with acc {metrics['accuracy']:.4f}")

        weights = torch.load(f"{save_dir}/best_model.pth", map_location=self.device)
        self.model.load_state_dict(weights)
        with torch.no_grad():
            eval_gt, eval_pred, eval_logit, eval_metrics = self.evaluate(split="test")

        print("==== Test Results ====")
        print(f"Test Loss: {eval_metrics['loss']:.4f}, Test Acc: {eval_metrics['accuracy']:.4f}, ")

    def save_predictions(self, eval_gt, eval_pred, eval_logit, save_dir, split):
        out_dict = {"pred_label": eval_pred, "gt_label": eval_gt}
        df_test_preds = pd.DataFrame(out_dict)
        df_test_preds.to_csv(f"{save_dir}/{split}_predictions.csv", index=False)
        return df_test_preds
