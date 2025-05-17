import torch

def training_loop():
    input = torch.tensor([1.0]).view(1, 1)

    model = torch.nn.Linear(1, 1)

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params)

    for i in range(3):
        optimizer.zero_grad()
        if i != 1:
            output = model(input)
            loss = output.sum()
            loss.backward()

        optimizer.step()
        print("step", optimizer.state[params[0]]["step"])

compiled_training_loop = torch._dynamo.optimize("eager", save_config=False)(training_loop)

print("expected in eager:")
training_loop()

print("what actually happens after dynamo:")
compiled_training_loop()
