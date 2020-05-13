SEGMAP_COLORS = [(0, 0, 0), (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
          (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180),
          (0, 0, 128), (128, 128, 128), (255, 255, 255), (115, 12, 37), (30, 90, 37), (127, 112, 12), (0, 65, 100),
          (122, 65, 24), (72, 15, 90), (35, 120, 120), (120, 25, 115), (105, 122, 30), (125, 95, 95), (0, 64, 64),
          (115, 95, 127), (85, 55, 20), (127, 125, 100), (64, 0, 0), (85, 127, 97), (64, 64, 0), (127, 107, 90),
          (0, 0, 64), (64, 64, 64)]

# Initiate settings (for testing purposes)
DEFAULT_SETTINGS = {
    # General settings
    'N_COLORS': 3,
    'COLORS_RESIZE': 128,
    'USE_CENTER': False,
    'UPLOAD_PATH': 'uploads',
    'ALLOWED_EXTENSIONS': set(['png', 'jpg', 'jpeg']),
    'REMOVE_BACKGROUND': True,
    'IMG_SIZE': 128,

    # Gender classifier settings
    'GENDER_BUCKETS': {0: 'female', 1: 'male'},

    # Age classifier settings
    'AGE_BUCKETS': {0: '10-14', 1: '15-19', 2: '20-24', 3: '25-29', 4: '30-34', 5: '35-39', 6: '40-44', 7: '45-49', 8: '50-54', 9: '55-59', 10: '60+'}, # 11 class model

    # Body extractor settings
    'BodyExtractor': {
        # Body detection settings
        'MIN_PROB': 40,
        'IMG_SIZE': 128,
        'UPPER_TO_LOWER_RATIO': 0.45,
        'SCALE_X': 0.5,
        'SCALE_Y': 0.5,

         # Face detection settings
        'FACE_SCALE_FACTOR': 1.03,
        'FACE_MIN_NEIGHBOURS': 20,
        'FACE_MIN_SIZE': (128, 128),

        # Eye detection settings
        'EYES_SCALE_FACTOR': 1.05,
        'EYES_MIN_NEIGHBOURS': 6
    },

    # Background remover settings
    'BackgroundRemover': {
        'MASK_COLOR': [255, 255, 255],
        'MASK_DILATE_ITER': 15,
        'MASK_ERODE_ITER': 25,
        'MASK_BLUR': 21,
        'MASK_THRESHOLD': 0.9,
        'MODEL_INPUT_SIZE': (150, 150)
    }
}
