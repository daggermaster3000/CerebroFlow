import PySimpleGUI as sg

# Define layouts for the different tabs
layout_tab1 = [
    [sg.Text("Content of Tab 1")],
    [sg.Button("Button 1")]
]

layout_tab2 = [
    [sg.Text("Content of Tab 2")],
    [sg.Button("Button 2")]
]

layout_tab3 = [
    [sg.Text("Content of Tab 3")],
    [sg.Button("Button 3")]
]

# Create a tab group with the defined layouts
tab_group_layout = [
    [sg.TabGroup([
        [sg.Tab("Tab 1", layout_tab1)],
        [sg.Tab("Tab 2", layout_tab2)],
        [sg.Tab("Tab 3", layout_tab3)]
    ])],
    [sg.Button("Exit")]
]

window = sg.Window("Tabs Example", tab_group_layout)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == "Exit":
        break

window.close()
