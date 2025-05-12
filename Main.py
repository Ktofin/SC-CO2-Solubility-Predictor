import flet as ft
from catboost import CatBoostRegressor
from CDK_pywrapper import CDK
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
import tempfile
import os
import sys

# === Инициализация CDK и модели ===
cdk_2 = CDK(ignore_3D=False)
cdk_name = list(pd.read_csv("CDK_DES.csv")['Name'])
model = CatBoostRegressor()
model.load_model("catboost_smiles_model.cbm")

def get_feature_vector(smiles, T, P, Tm, dHvap, rho, dG):
    smiles_list = [smiles, "O"]
    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles_list]
    uncharger = rdMolStandardize.Uncharger()
    mols = [uncharger.uncharge(mol) for mol in mols]
    for mol in mols:
        AllChem.EmbedMolecule(mol, randomSeed=42)
    descriptors = cdk_2.calculate(mols).head(1)
    desc_vector = descriptors.loc[:, cdk_name].values[0].tolist()
    return [T, P, Tm, dHvap, rho, dG] + desc_vector

def main(page: ft.Page):
    page.title = "SC-CO2 Solubility Predictor"
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 30
    page.window_width = 800
    page.window_height = 720

    smiles_field = ft.TextField(label="SMILES (обязательно)", expand=True, tooltip="Структура соединения в формате SMILES")
    T_field = ft.TextField(label="Температура (K, обязательно)", keyboard_type=ft.KeyboardType.NUMBER, tooltip="Температура процесса в Кельвинах")
    P_field = ft.TextField(label="Давление (bar, обязательно)", keyboard_type=ft.KeyboardType.NUMBER, tooltip="Давление процесса в барах")
    Tm_field = ft.TextField(label="Температура плавления (K)", keyboard_type=ft.KeyboardType.NUMBER, tooltip="Температура плавления вещества")
    dHvap_field = ft.TextField(label="Энтальпия испарения (кДж/моль)", keyboard_type=ft.KeyboardType.NUMBER, tooltip="Энтальпия испарения вещества")
    rho_field = ft.TextField(label="Плотность CO2 (г/см³)", keyboard_type=ft.KeyboardType.NUMBER, tooltip="Плотность CO2 при заданных условиях")
    dG_field = ft.TextField(label="Энергия сольватации (кДж/моль)", keyboard_type=ft.KeyboardType.NUMBER, tooltip="Гиббсовая энергия сольватации")

    result = ft.Text("", size=16, selectable=True)
    image = ft.Image(src="", width=250, height=250, visible=False)

    def predict_solubility(e):
        try:
            if not smiles_field.value.strip() or not T_field.value or not P_field.value:
                result.value = "\u274C Пожалуйста, заполните обязательные поля: SMILES, Температура и Давление."
                image.visible = False
            else:
                Tm = float(Tm_field.value) if Tm_field.value else 0.0
                dHvap = float(dHvap_field.value) if dHvap_field.value else 0.0
                rho = float(rho_field.value) if rho_field.value else 0.0
                dG = float(dG_field.value) if dG_field.value else 0.0

                smiles = smiles_field.value.strip()
                vector = get_feature_vector(smiles, float(T_field.value), float(P_field.value), Tm, dHvap, rho, dG)
                log_y2 = model.predict([vector])[0]
                y2 = 10 ** log_y2
                result.value = f"\u2705 log(y₂) = {log_y2:.4f}\nРастворимость y₂ ≈ {y2:.5f} г/г CO₂"

                # Отрисовка молекулы и временное сохранение
                mol = Chem.MolFromSmiles(smiles)
                img = Draw.MolToImage(mol, size=(300, 300))
                temp_path = os.path.join(tempfile.gettempdir(), "mol.png")
                img.save(temp_path)
                image.src = temp_path
                image.visible = True

        except Exception as ex:
            result.value = f"\u274C Ошибка: {str(ex)}"
            image.visible = False
        page.update()

    page.add(
        ft.Column([
            ft.Row([
                ft.Text("🧪", size=28),
                ft.Text("SC-CO₂ Solubility Predictor", size=26, weight="bold")
            ], alignment=ft.MainAxisAlignment.CENTER),

            ft.Text("Единицы измерения: растворимость выражается в граммах вещества на грамм CO₂ (г/г)", size=14, italic=True, text_align="center"),
            ft.Divider(height=20),

            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("Ввод данных", size=18, weight="bold"),
                        ft.ResponsiveRow([
                            ft.Column([smiles_field], col=12),
                            ft.Column([T_field, Tm_field, dHvap_field], col=6),
                            ft.Column([P_field, rho_field, dG_field], col=6),
                        ])
                    ]),
                    bgcolor=ft.colors.BLUE_GREY_50,
                    border_radius=10,
                    padding=20
                )
            ),

            ft.Container(
                content=ft.ElevatedButton("\u2696\ufe0f Рассчитать", on_click=predict_solubility, style=ft.ButtonStyle(bgcolor=ft.colors.BLUE, color=ft.colors.WHITE)),
                padding=ft.padding.only(top=20, bottom=10)
            ),

            ft.Row([
                ft.Container(
                    content=result,
                    padding=15,
                    border_radius=8,
                    border=ft.border.all(1, ft.colors.GREY_300),
                    bgcolor=ft.colors.WHITE,
                    expand=1
                ),
                ft.Container(
                    content=image,
                    padding=10,
                    expand=0
                )
            ], alignment=ft.MainAxisAlignment.CENTER)
        ], width=600, alignment=ft.MainAxisAlignment.CENTER)
    )

ft.app(target=main)

