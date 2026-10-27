import albumentations as A
import cv2

def build_train_augmentation(seed: int | None = None) -> A.Compose:
    """
    Pipeline de augmentation para entrenamiento.
    Aplica transformaciones que simulan variaciones reales entre fotos de una misma
    persona sin alterar su identidad.
    
    Args:
        seed: Fija la aleatoriedad (reproducibilidad).

    Returns:
        A.Compose: Pipeline de albumentations.
    """
    return A.Compose(
        [
            # Flip horizontal
            A.HorizontalFlip(p=0.5),
            # Giros, corrimientos y cambios de escala pequeños
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.04, 0.04), rotate=(-8, 8), border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            # Iluminación variable
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            # Balance de blancos
            A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            # Blanco y negro
            A.ToGray(p=0.05),
            # Desenfoque de foco o de movimiento
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 7))], p=0.2),
            # Ruido de sensor con poca luz
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            # Compresión
            A.ImageCompression(quality_range=(45, 95), p=0.3),
            # Oclusiones parciales
            A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.10, 0.25), hole_width_range=(0.10, 0.25), fill=0, p=0.25),
        ],
        seed=seed
    )
