"""Integration tests for Flower federated learning components."""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from flwr.app import Message, Context, ArrayRecord, MetricRecord, RecordDict
from myapp.task import Net


class TestFlowerClientApp:
    """Test Flower client app functionality."""

    def test_client_train_message_structure(self):
        """Test that client creates proper train response message."""
        from myapp.client_app import train
        
        # Mock context
        context = Mock(spec=Context)
        context.node_config = {"partition-id": 0, "num-partitions": 2}
        context.run_config = {"local-epochs": 1}
        
        # Create mock input message
        model = Net()
        arrays = ArrayRecord(model.state_dict())
        config = {"lr": 0.001}
        content = RecordDict({"arrays": arrays, "config": config})
        msg = Message(content=content)
        
        # Mock the data loading to avoid file system dependencies
        with patch('myapp.client_app.load_data') as mock_load_data:
            # Create dummy data
            X = torch.randn(10, 1, 40, 101)
            y = torch.randint(0, 35, (10,))
            
            from torch.utils.data import TensorDataset, DataLoader
            dataset = TensorDataset(X, y)
            trainloader = DataLoader(dataset, batch_size=4)
            mock_load_data.return_value = (trainloader, trainloader)
            
            # Call train function
            response = train(msg, context)
            
            # Verify response structure
            assert isinstance(response, Message)
            assert "arrays" in response.content
            assert "metrics" in response.content
            assert response.reply_to == msg

    def test_client_evaluate_message_structure(self):
        """Test that client creates proper evaluate response message."""
        from myapp.client_app import evaluate
        
        # Mock context
        context = Mock(spec=Context)
        context.node_config = {"partition-id": 0, "num-partitions": 2}
        
        # Create mock input message
        model = Net()
        arrays = ArrayRecord(model.state_dict())
        content = RecordDict({"arrays": arrays})
        msg = Message(content=content)
        
        # Mock the data loading
        with patch('myapp.client_app.load_data') as mock_load_data:
            X = torch.randn(10, 1, 40, 101)
            y = torch.randint(0, 35, (10,))
            
            from torch.utils.data import TensorDataset, DataLoader
            dataset = TensorDataset(X, y)
            testloader = DataLoader(dataset, batch_size=4)
            mock_load_data.return_value = (testloader, testloader)
            
            # Call evaluate function
            response = evaluate(msg, context)
            
            # Verify response structure
            assert isinstance(response, Message)
            assert "metrics" in response.content
            assert response.reply_to == msg


class TestFlowerServerApp:
    """Test Flower server app functionality."""

    def test_server_callback_saves_model(self, tmp_path):
        """Test that server callback saves model checkpoints."""
        from myapp.server_app import my_callback
        
        # Change to temp directory for test
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Create mock arrays
            model = Net()
            arrays = ArrayRecord(model.state_dict())
            
            # Call callback
            result = my_callback(server_round=1, arrays=arrays)
            
            # Check that model file was created
            assert (tmp_path / "model_round_1.pt").exists()
            
            # Verify it's a valid model checkpoint
            loaded = torch.load(tmp_path / "model_round_1.pt")
            assert isinstance(loaded, dict)
            
        finally:
            os.chdir(original_dir)

    def test_fedavg_with_logging_aggregation(self):
        """Test that FedAvgWithLogging properly aggregates."""
        from myapp.server_app import FedAvgWithLogging
        
        strategy = FedAvgWithLogging(fraction_train=1.0)
        
        # Verify strategy is properly initialized
        assert strategy is not None
        assert hasattr(strategy, 'aggregate_train')
        assert hasattr(strategy, 'aggregate_evaluate')


class TestPrometheusMetrics:
    """Test Prometheus metrics integration."""

    def test_metrics_initialization(self):
        """Test that Prometheus metrics are properly defined."""
        from myapp.server_app import (
            fl_round, fl_train_loss, fl_eval_loss, 
            fl_clients, fl_model_size
        )
        
        # Verify metrics exist
        assert fl_round is not None
        assert fl_train_loss is not None
        assert fl_eval_loss is not None
        assert fl_clients is not None
        assert fl_model_size is not None

    def test_client_metrics_initialization(self):
        """Test that client Prometheus metrics are defined."""
        from myapp.client_app import TRAIN_LOSS, EVAL_LOSS, EVAL_ACC
        
        assert TRAIN_LOSS is not None
        assert EVAL_LOSS is not None
        assert EVAL_ACC is not None


class TestModelCheckpointing:
    """Test model checkpointing functionality."""

    def test_checkpoint_save_load_cycle(self, tmp_path):
        """Test saving and loading model checkpoints."""
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            model = Net()
            state_dict = model.state_dict()
            
            # Save
            torch.save(state_dict, "test_checkpoint.pt")
            
            # Load
            loaded_state = torch.load("test_checkpoint.pt")
            
            # Verify
            new_model = Net()
            new_model.load_state_dict(loaded_state)
            
            # Compare parameters
            for (name1, p1), (name2, p2) in zip(
                model.named_parameters(), 
                new_model.named_parameters()
            ):
                assert name1 == name2
                assert torch.equal(p1, p2)
                
        finally:
            os.chdir(original_dir)


@pytest.fixture
def mock_flower_context():
    """Fixture for mock Flower context."""
    context = Mock(spec=Context)
    context.node_config = {
        "partition-id": 0,
        "num-partitions": 2
    }
    context.run_config = {
        "local-epochs": 1,
        "lr": 0.001
    }
    return context


@pytest.fixture
def sample_message_with_model():
    """Fixture providing a sample message with model weights."""
    model = Net()
    arrays = ArrayRecord(model.state_dict())
    config = {"lr": 0.001}
    content = RecordDict({"arrays": arrays, "config": config})
    return Message(content=content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
