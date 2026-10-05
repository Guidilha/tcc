import os
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class HyperNNTensorBoardLogger:
    """
    Logger dedicado para registrar métricas e distribuições do HyperNN no TensorBoard.
    Organiza os experimentos na estrutura: runs/<dataset>/<strategy>/seed_<seed>
    permitindo que o TensorBoard agrupe automaticamente as seeds e exiba média + desvio padrão.
    """

    def __init__(self, log_dir="runs", enabled=True):
        self.log_dir = log_dir
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)

    def log_experiment_result(self, dataset_name, strategy_name, model, metrics_dict, seed=42, step=0):
        """
        Registra os resultados finais da avaliação do modelo para um dataset, estratégia e seed específicos.

        :param dataset_name: Nome do dataset (ex: 'iris', 'wine')
        :param strategy_name: Nome da estratégia (ex: 'som', 'kmeans', 'dbscan', 'random')
        :param model: Instância do melhor modelo avaliado
        :param metrics_dict: Dicionário com as métricas de teste (f1_score, tempos, nboxes)
        :param seed: Número da semente aleatória (ex: 42, 43, 44)
        :param step: Passo/época do log (padrão 0 para métrica estática final)
        """
        if not self.enabled:
            return

        # Estrutura de diretório hierárquica: runs/iris/random/seed_42
        run_id = os.path.join(dataset_name, strategy_name, f"seed_{seed}")
        log_path = os.path.join(self.log_dir, run_id)
        writer = SummaryWriter(log_dir=log_path)

        try:
            # 1. Registra Métricas Escalares de Desempenho
            for metric_name, value in metrics_dict.items():
                if isinstance(value, dict):
                    for sub_k, sub_v in value.items():
                        try:
                            writer.add_scalar(f"Metrics/{metric_name}_{sub_k}", float(sub_v), step)
                        except (ValueError, TypeError):
                            pass
                else:
                    try:
                        writer.add_scalar(f"Metrics/{metric_name}", float(value), step)
                    except (ValueError, TypeError):
                        pass

            # 2. Registra Tempo de Inicialização do Modelo
            init_time = getattr(model, "init_time", 0.0)
            writer.add_scalar("Timing/initialization_time", float(init_time), step)

            # 3. Registra Número de Hipercaixas Ativas Finais
            if hasattr(model, "get_num_boxes"):
                writer.add_scalar("Hyperboxes/final_nboxes", float(model.get_num_boxes()), step)

            # 4. Registra Histogramas das Hipercaixas (mins e lengths)
            if hasattr(model, "clf") and hasattr(model.clf, "model") and hasattr(model.clf.model, "block"):
                block = model.clf.model.block
                with torch.no_grad():
                    writer.add_histogram("Hyperboxes/Mins_Distribution", block.mins.cpu(), step)
                    writer.add_histogram("Hyperboxes/Lengths_Distribution", block.length.cpu(), step)

            writer.flush()
        finally:
            writer.close()
