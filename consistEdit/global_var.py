# global_vars.py
class GlobalVars:
    IS_INVERSE = False
    TOTAL_STEPS = 0
    NUM_ATT_LAYERS = 0
    NUM_ATT_2_LAYERS = 0
    CUR_STEP = 0
    CUR_ATT_LAYER = 0
    CUR_ATT_2_LAYER = 0
    ORDER = 0
    attn_controller = None

    TEST_V_STEP = 0
    TEST_QK_STEP = 0
    NO_V = False
    TEXT_LENGTH = 333
    WIDTH = 1024//8//2
    HEIGHT = 1024//8//2

    GENERATE_MASK = False

    MASK= None
    TOKEN_IDS = None
    MASK_OUTPUT_PATH = None
    REGENERATE_MASK = False
    SAVE_MASK_TRIGGER = False
    MAP_SAVER = {}

    @classmethod
    def get_attn_controller(cls):
        """Lazy initialization of attention controller to avoid circular import"""
        if cls.attn_controller is None:
            from consistEdit.attention_map_controller import AttentionMapController
            cls.attn_controller = AttentionMapController("sd3_mask_output", thres=0.1)
        return cls.attn_controller

    @classmethod
    def reset_global_vars(cls):
        cls.SAVE_MASK_TRIGGER = False
        cls.IS_INVERSE = False
        cls.TOTAL_STEPS = 0
        cls.CUR_STEP = 0
        cls.CUR_ATT_LAYER = 0
        cls.CUR_ATT_2_LAYER = 0
        cls.ORDER = 0
        from consistEdit.attention_map_controller import AttentionMapController
        cls.attn_controller = AttentionMapController("sd3_mask_output", thres=0.1)