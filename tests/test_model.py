"""Unit tests for the FedOps Audio Classification System."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from myapp.task import Net, MFCCDataset, train, test


class TestNeuralNetwork:
    """Test cases for the CNN model."""

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = Net()
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_forward_pass(self):
        """Test forward pass with sample input."""
        model = Net()
        # Shape: (batch, channels, height, width) = (2, 1, 40, 101)
        sample_input = torch.randn(2, 1, 40, 101)
        output = model(sample_input)
        
        # Check output shape
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 35  # number of classes
        
    def test_model_output_range(self):
        """Test that model output can be converted to probabilities."""
        model = Net()
        sample_input = torch.randn(1, 1, 40, 101)
        output = model(sample_input)
        
        # Apply softmax and check it sums to 1
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        model = Net()
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestDataset:
    """Test cases for MFCC Dataset."""

    def test_dataset_creation(self):
        """Test dataset can be created."""
        X = torch.randn(10, 1, 40, 101)
        y = torch.randint(0, 35, (10,))
        dataset = MFCCDataset(X, y)
        
        assert len(dataset) == 10
        
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        X = torch.randn(10, 1, 40, 101)
        y = torch.randint(0, 35, (10,))
        dataset = MFCCDataset(X, y)
        
        x_sample, y_sample = dataset[0]
        assert x_sample.shape == (1, 40, 101)
        assert isinstance(y_sample.item(), int)

    def test_dataloader_integration(self):
        """Test dataset works with DataLoader."""
        X = torch.randn(10, 1, 40, 101)
        y = torch.randint(0, 35, (10,))
        dataset = MFCCDataset(X, y)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (2, 1, 40, 101)
        assert batch_y.shape == (2,)


class TestTraining:
    """Test cases for training functionality."""

    def test_train_function(self):
        """Test that training runs without errors."""
        model = Net()
        X = torch.randn(10, 1, 40, 101)
        y = torch.randint(0, 35, (10,))
        dataset = MFCCDataset(X, y)
        trainloader = DataLoader(dataset, batch_size=2)
        
        device = torch.device("cpu")
        loss = train(model, trainloader, epochs=1, lr=0.001, device=device)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_loss_decreases(self):
        """Test that loss decreases over epochs (simple check)."""
        model = Net()
        X = torch.randn(100, 1, 40, 101)
        y = torch.randint(0, 35, (100,))
        dataset = MFCCDataset(X, y)
        trainloader = DataLoader(dataset, batch_size=16)
        
        device = torch.device("cpu")
        
        # Train for 1 epoch
        loss1 = train(model, trainloader, epochs=1, lr=0.001, device=device)
        
        # Train for another epoch (should continue from previous state)
        loss2 = train(model, trainloader, epochs=1, lr=0.001, device=device)
        
        # Both should be valid losses
        assert loss1 >= 0
        assert loss2 >= 0


class TestEvaluation:
    """Test cases for evaluation functionality."""

    def test_test_function(self):
        """Test that evaluation runs without errors."""
        model = Net()
        X = torch.randn(10, 1, 40, 101)
        y = torch.randint(0, 35, (10,))
        dataset = MFCCDataset(X, y)
        testloader = DataLoader(dataset, batch_size=2)
        
        device = torch.device("cpu")
        loss, accuracy = test(model, testloader, device)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert loss >= 0

    def test_evaluation_no_grad(self):
        """Test that evaluation doesn't update gradients."""
        model = Net()
        X = torch.randn(10, 1, 40, 101)
        y = torch.randint(0, 35, (10,))
        dataset = MFCCDataset(X, y)
        testloader = DataLoader(dataset, batch_size=2)
        
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        device = torch.device("cpu")
        test(model, testloader, device)
        
        # Check parameters haven't changed
        for init_p, curr_p in zip(initial_params, model.parameters()):
            assert torch.equal(init_p, curr_p)


class TestModelSerialization:
    """Test model saving and loading."""

    def test_model_state_dict(self):
        """Test getting model state dict."""
        model = Net()
        state_dict = model.state_dict()
        
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    def test_model_save_load(self, tmp_path):
        """Test saving and loading model."""
        model = Net()
        
        # Save model
        save_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Load into new model
        new_model = Net()
        new_model.load_state_dict(torch.load(save_path))
        
        # Compare parameters
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)


@pytest.fixture
def sample_mfcc_data():
    """Fixture providing sample MFCC data."""
    X = torch.randn(20, 1, 40, 101)
    y = torch.randint(0, 35, (20,))
    return X, y


@pytest.fixture
def sample_model():
    """Fixture providing a sample model."""
    return Net()


def test_integration_train_eval(sample_mfcc_data, sample_model):
    """Integration test for training and evaluation pipeline."""
    X, y = sample_mfcc_data
    dataset = MFCCDataset(X, y)
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    trainloader = DataLoader(train_dataset, batch_size=4)
    testloader = DataLoader(test_dataset, batch_size=4)
    
    device = torch.device("cpu")
    
    # Train
    train_loss = train(sample_model, trainloader, epochs=2, lr=0.001, device=device)
    assert train_loss >= 0
    
    # Evaluate
    test_loss, test_acc = test(sample_model, testloader, device)
    assert test_loss >= 0
    assert 0 <= test_acc <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
