import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def z_function(x_coord, y_coord):
    return x_coord * np.sin(y_coord)


x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X1, Y1 = np.meshgrid(x_vals, y_vals)
Z_vals = z_function(X1, Y1)

X = np.column_stack((X1.ravel(), Y1.ravel()))
Z = Z_vals.ravel()

x_train, x_test, y_train, y_test = train_test_split(X, Z, test_size=0.1)

scaler = StandardScaler()
x_train_t = torch.FloatTensor(scaler.fit_transform(x_train))
x_test_t = torch.FloatTensor(scaler.transform(x_test))
y_train_t = torch.FloatTensor(y_train)
y_test_t = torch.FloatTensor(y_test)


class FeedNet(nn.Module):
    def __init__(self, hidden_layers):
        super(FeedNet, self).__init__()
        input_size = 2
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, input_data):
        return self.network(input_data)


class CascadeNet(nn.Module):
    def __init__(self, hidden_layers):
        super(CascadeNet, self).__init__()
        self.hidden_layers = nn.ModuleList()
        input_size = 2
        prev_size = input_size
        for hidden_size in hidden_layers:
            self.hidden_layers.append(nn.Linear(prev_size + input_size, hidden_size))
            prev_size = hidden_size
        self.output = nn.Linear(prev_size + input_size, 1)

    def forward(self, input_data):
        out = input_data
        for layer in self.hidden_layers:
            out = torch.cat([input_data, out], dim=1)
            out = torch.tanh(layer(out))
        out = torch.cat([input_data, out], dim=1)
        return self.output(out)


class ElmanNet(nn.Module):
    def __init__(self, hidden_sizes, output_size=1):
        super(ElmanNet, self).__init__()
        self.hidden_sizes = hidden_sizes
        input_size = 2
        self.i2h = nn.ModuleList([nn.Linear(input_size + hidden_size, hidden_size) for hidden_size in hidden_sizes])
        self.h2o = nn.Linear(hidden_sizes[-1], output_size)
        self.tanh = nn.Tanh()

    def forward(self, input_data, hidden_states):
        combined = torch.cat((input_data, hidden_states[0]), 1)
        hidden_next = []
        for i, i2h_layer in enumerate(self.i2h):
            hidden_i = self.tanh(i2h_layer(combined))
            combined = torch.cat((input_data, hidden_i), 1)
            hidden_next.append(hidden_i)
        output = self.h2o(hidden_next[-1])
        return output, hidden_next

    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, hidden_size) for hidden_size in self.hidden_sizes]


def train_model(model_net_instance, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_net_instance.parameters(), lr=lr)
    loss_curve_values = []

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    model_net_instance.apply(init_weights)

    for epoch in range(epochs):
        model_net_instance.train()

        if isinstance(model_net_instance, ElmanNet):
            hidden = model_net_instance.init_hidden(x_train_t.size(0))
            output, hidden = model_net_instance(x_train_t, hidden)
        else:
            output = model_net_instance(x_train_t)

        loss = criterion(output, y_train_t.unsqueeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_curve_values.append(loss.item())
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    model_net_instance.eval()
    with torch.no_grad():
        if isinstance(model_net_instance, ElmanNet):
            hidden = model_net_instance.init_hidden(x_test_t.size(0))
            test_predictions, hidden = model_net_instance(x_test_t, hidden)
        else:
            test_predictions = model_net_instance(x_test_t)
        mre = torch.mean(torch.abs((y_test_t - test_predictions.squeeze()) / y_test_t)).item()

    return test_predictions, mre, loss_curve_values


neural_network_config = {
    '1 layer (10) FN': {'model': FeedNet, 'params': {'hidden_layers': [10]}},
    '1 layer (20) FN': {'model': FeedNet, 'params': {'hidden_layers': [20]}},
    '1 layer (20) CN': {'model': CascadeNet, 'params': {'hidden_layers': [20]}},
    '2 layers (10 each) CN': {'model': CascadeNet, 'params': {'hidden_layers': [10, 10]}},
    '1 layer (15) EN': {'model': ElmanNet, 'params': {'hidden_sizes': [15]}},
    '3 layers (5 each) EN': {'model': ElmanNet, 'params': {'hidden_sizes': [5, 5, 5]}},
}

for choice, config in neural_network_config.items():
    print(f"\nTraining configuration {choice}")
    net_instance = config['model'](**config['params'])
    test_predictions_iter, mre_iter, loss_curve = train_model(net_instance, epochs=6000, lr=0.01)

    print(f"MRE: {mre_iter:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, test_predictions_iter.numpy().squeeze(), color='red')
    plt.plot([-3, 3], [-3, 3], color='blue', linestyle='--')
    plt.xlabel('Actual Values (R)')
    plt.ylabel('Predicted Values (P)')
    plt.title(f'Network Type: {config["model"].__name__}, Configuration: {choice}')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(loss_curve, color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for: {config["model"].__name__}, Configuration: {choice}')
    plt.ylim(0, 1)
    plt.grid()

    plt.tight_layout()
    plt.show()
