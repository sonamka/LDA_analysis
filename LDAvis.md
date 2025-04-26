```python
#After conducting the LDA analysis, use joblib to save the model so if you need to restart kernel, you don't need to recreate everything.
import joblib

# Load the objects
best_lda_model = joblib.load('lda_model.pkl')
new_matrix = joblib.load('new_matrix.pkl')
vectorizer = joblib.load('vectorizer.pkl')
```


```python
import pyLDAvis
```


```python
pip show pyLDAvis
```

    Name: pyLDAvisNote: you may need to restart the kernel to use updated packages.
    
    Version: 3.4.1
    Summary: Interactive topic model visualization. Port of the R package.
    Home-page: https://github.com/bmabey/pyLDAvis
    Author: Ben Mabey
    Author-email: ben@benmabey.com
    License: BSD-3-Clause
    Location: C:\Users\sahluwal\AppData\Local\anaconda3\Lib\site-packages
    Requires: funcy, gensim, jinja2, joblib, numexpr, numpy, pandas, scikit-learn, scipy, setuptools
    Required-by: 
    


```python
import pyLDAvis.lda_model
```


```python
from pyLDAvis.lda_model import prepare
```


```python
pyLDAvis.enable_notebook()
vis = pyLDAvis.lda_model.prepare(best_lda_model, new_matrix, vectorizer)
```


```python
pyLDAvis.save_html(vis, 'lda_visualization.html')
```


```python

```
