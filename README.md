# OCT Retina AI Analyzer (PatchBridgeNet GUI)

## 1. Colab me model banao

1. `PatchBridgeNet_Colab_Pipeline.ipynb` Colab me kholo (GPU runtime).
2. Runtime > Restart session, phir Cell 1 se last cell tak sab chalao.
   Features aur INCA Drive cache se load honge, isliye jaldi ho jayega.
3. Last cell `deployment.zip` download karega.

## 2. Local folder

```
oct_gui/
    app.py
    pbn_core.py
    requirements.txt
    deployment/            <- deployment.zip yahan unzip karo
        retina_7class.pkl
        retina_binary.pkl
        cnn_weights.pt
        metrics.json
        training_info.json
        confusion_matrix_7class.png
        confusion_matrix_binary.png
```

`deployment.zip` ke andar files seedhe root pe hain, isliye pehle `deployment` folder banao aur zip usi ke andar extract karo.

`app.py` ke saath wala `pbn_core.py` wahi hona chahiye jo notebook ke Cell 4 me hai (zip me bhi ek copy aati hai). Dono alag hue to percentages galat aa sakte hain.

## 3. Run

```
python -m venv venv
venv\Scripts\activate          (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Browser me `http://localhost:8501` khulega. GPU nahi hai to CPU pe chalega; ek image me kuch second lagte hain, Grad-CAM me thoda zyada.

scikit-learn ki zaroorat nahi hai, pickle files sirf NumPy arrays hain.
