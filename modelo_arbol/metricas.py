def calcular_metricas(y, y_pred):
    tp = sum((y == True) & (y_pred == True))
    tn = sum((y == False) & (y_pred == False))
    fp = sum((y == False) & (y_pred == True))
    fn = sum((y == True) & (y_pred == False))

    precision = (tp + tn) / len(y)
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

    from core.logging_config import logger
    logger.info(f"Precisión global: {precision:.2%}")
    logger.info(f"Precisión (Pico): {precision_pos:.2%}")
    logger.info(f"Recall (Pico): {recall_pos:.2%}")
    logger.info(f"F1-score: {f1:.2%}")
