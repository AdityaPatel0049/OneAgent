from backend_auto_weights import auto_weights
from domain_analyzers import analyze_financial, analyze_medical, analyze_generic

t = (
    "For discomfort, give your child acetaminophen (Tylenol) or ibuprofen (Advil, Motrin). "
    "There are two medications to help relieve fevers. Paracetamol 1000 mg represents the first choice for treatment of fever."
)

auto = auto_weights(t, agent="General")
print("meta:", auto["meta"]) 
print("weights_fin:", auto["weights_fin"]) 
print("weights_med:", auto["weights_med"]) 
print("weights_gen:", auto["weights_gen"]) 
print("financial:", analyze_financial(t, weights=auto["weights_fin"]))
print("medical:", analyze_medical(t, weights=auto["weights_med"]))
print("generic:", analyze_generic(t, weights=auto["weights_gen"]))
