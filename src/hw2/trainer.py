class ModelTrainer(object):
    def __init__(self, model, criterion, optimizer, data):
        self.criterion = criterion
        self.optimizer = optimizer
        self.data = data
        self.model = model
        self.stats = []

    def train_epoch(self, device="cuda"):
        self.model.train()
        iter_count = 0
        total_loss = 0
        self.stats.append({
            "loss": [],
            "gradient": []
        })
        for input, target in self.data.train_iterator():
            input = input.to(device)
            target = target.to(device)
            output = self.model.predict(input, device=device)
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.stats[-1]["loss"].append(loss.item())
            self.stats[-1]["gradient"].append(self.model.get_gradient_norm())
            total_loss += loss.item()
            iter_count += 1
        return total_loss / iter_count

    def evaluate_epoch(self, device="cuda"):
        self.model.eval()
        iter_count = 0
        total_loss = 0
        for input, target in self.data.test_iterator():
            input = input.to(device)
            target = target.to(device)
            output = self.model.predict(input, device=device)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            iter_count += 1
        return total_loss / iter_count

    def train(self, n_epochs, device="cuda"):
        self.stats = []
        for epoch in range(0, n_epochs):
            train_loss = self.train_epoch(device=device)
            test_loss = self.evaluate_epoch(device=device)
            print("Epoch {epoch}: Train Loss = {train_loss} Test Loss = {test_loss}".format(
                epoch=epoch, train_loss=train_loss, test_loss=test_loss))
