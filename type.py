import numpy as np
import sys

def print_dict_structure(obj, indent=0, max_depth=5):
    """Afișează structura unui obiect (dict, array, etc.) în format imbricat."""
    prefix = "  " * indent
    
    if indent > max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    if isinstance(obj, dict):
        print(f"{prefix}Dict cu {len(obj)} chei:")
        for k, v in obj.items():
            print(f"{prefix}  '{k}':", end=" ")
            if isinstance(v, np.ndarray):
                info = f"numpy array, shape={v.shape}, dtype={v.dtype}"
                if v.size > 0 and np.issubdtype(v.dtype, np.number):
                    try:
                        info += f", min={v.min():.4f}, max={v.max():.4f}"
                    except:
                        pass
                print(info)
            elif isinstance(v, dict):
                print()
                print_dict_structure(v, indent + 2, max_depth)
            elif isinstance(v, (list, tuple)):
                print(f"{type(v).__name__} cu {len(v)} elemente")
                if len(v) > 0:
                    print_dict_structure(v[0], indent + 2, max_depth)
            else:
                print(f"{type(v).__name__} = {v if len(str(v)) < 50 else str(v)[:50] + '...'}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} cu {len(obj)} elemente:")
        for i, item in enumerate(obj[:3]):  # Arata primele 3 elemente
            print(f"{prefix}  [{i}]:", end=" ")
            if isinstance(item, np.ndarray):
                info = f"numpy array, shape={item.shape}, dtype={item.dtype}"
                if item.size > 0 and np.issubdtype(item.dtype, np.number):
                    try:
                        info += f", min={item.min():.4f}, max={item.max():.4f}"
                    except:
                        pass
                print(info)
            else:
                print(f"{type(item).__name__}")

path = sys.argv[1] if len(sys.argv) > 1 else "P01_IMG1.npy"

data = np.load(path, allow_pickle=True)

print(f"Tip obiect: {type(data)}")
print(f"dtype:      {data.dtype}")

# Cazul cel mai comun: array simplu
if isinstance(data, np.ndarray):
    print(f"Shape:      {data.shape}")
    
    # Daca e 0-dimensional array cu dtype object, extragi elementul
    if data.shape == () and data.dtype == object:
        extracted = data.item()
        print(f"Tipul elementului: {type(extracted)}")
        if isinstance(extracted, dict):
            print(f"\nStructura detaliată a dictionarului:")
            print_dict_structure(extracted, indent=1)
        else:
            print(f"Conținut: {extracted}")
    # Daca e array numeric, calculeaza min/max
    elif np.issubdtype(data.dtype, np.number):
        print(f"Min / Max:  {data.min():.4f} / {data.max():.4f}")
        print(f"Valori unice (primele 20): {np.unique(data)[:20]}")
    else:
        print(f"Array cu dtype non-numeric: {data.dtype}")
        print(f"Conținut: {data}")

# Daca e un dict salvat cu allow_pickle (np.save cu dict)
elif isinstance(data, dict) or (hasattr(data, 'item') and isinstance(data.item(), dict)):
    d = data.item() if not isinstance(data, dict) else data
    print("\nStructura detaliată a dictionarului:")
    print_dict_structure(d, indent=0)

# Daca e array de obiecte (tuple/list salvat ca object array)
elif data.dtype == object:
    print(f"\nArray de tip object cu {data.shape} elemente:")
    item = data.item() if data.ndim == 0 else data[0]
    print(f"  Primul element: tip={type(item)}")
    if isinstance(item, (tuple, list)):
        for i, x in enumerate(item):
            if isinstance(x, np.ndarray):
                print(f"    [{i}]: shape={x.shape}, dtype={x.dtype}, min={x.min():.4f}, max={x.max():.4f}")
            else:
                print(f"    [{i}]: {type(x)} = {x}")