import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # Baslik ve Aciklama
    nb.cells.append(nbf.v4.new_markdown_cell("# Biyometrik Ses Analizi Projesi\n\nBu notebook, projeye ait tum kodlari tek bir dosyada toplamaktadir."))
    
    files_to_add = ['utils.py', 'inference.py', 'main.py', 'evaluation.py', 'enroll_user.py', 'app.py']
    
    for py_file in files_to_add:
        if os.path.exists(py_file):
            nb.cells.append(nbf.v4.new_markdown_cell(f"### Dosya: `{py_file}`"))
            with open(py_file, 'r', encoding='utf-8') as f:
                code = f.read()
            nb.cells.append(nbf.v4.new_code_cell(code))
        else:
            print(f"Uyari: {py_file} bulunamadi.")
            
    with open('OgrenciNo1_OgrenciNo2_Kod.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print("OgrenciNo1_OgrenciNo2_Kod.ipynb basariyla olusturuldu!")

if __name__ == "__main__":
    create_notebook()
