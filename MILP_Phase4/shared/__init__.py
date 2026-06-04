from shared.scip_features import (
    VAR_FEAT_DIM, 
    CON_FEAT_DIM,
    VAR_FEATURE_NAMES, 
    CON_FEATURE_NAMES,
    # حذفنا الأسماء الصغيرة والأسماء غير الموجودة
    extract_bipartite_graph,
)
from shared.shared_encoder import SharedEncoder, GATConvolution, Prenorm
# ملاحظة: build_shared_encoder تم حذفه من ملف shared_encoder.py حسب الـ Change Log الخاص بك