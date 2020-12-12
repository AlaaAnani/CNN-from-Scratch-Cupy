def graph_bar(x,y,
    graph_title,
    graph_file_name,
    x_title,
    y_title,
    graphs_folder_name
    ):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        text=y,
        textposition='auto'
    ))
    fig.update_layout(
        title=graph_title,\
        xaxis_title=x_title,\
        yaxis_title=y_title,\
        showlegend=False
    )
    fig.write_image(graphs_folder_name + '/' + graph_file_name + '.png')
    fig.write_html(graphs_folder_name + '/' + graph_file_name + '.html')
    if annotation is not None:
        for ann in annotation:
            fig.add_annotation(
                    x=ann[0],
                    y=ann[1],
                    xref="x",
                    yref="y",
                    text=ann[2],
                    showarrow=True,
                    font=dict(
                        family="Courier New, monospace",
                        size=13,
                        color="#000000"
                        ),
                    align="center",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    ax=20,
                    ay=-40,
                    bordercolor="#c7c7c7",
                    borderwidth=1,
                    borderpad=3,
                    bgcolor="#ff7f0e",
                    opacity=0.8
                    )