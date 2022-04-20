import pandas as pd

path = 'data/'

sections = pd.read_excel(path+'road_sections.xlsx')

first_axis = sections[sections.Axe == 1]
first_axis.to_excel('data/first_axis.xlsx', index=False)