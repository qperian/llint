methods = ['Vanilla', 'OrthoProj', 'OrthoCali', 'SFID', 'BEND-VLM', 'LLINT', 'GLINT']

ref_dataset = 'UTKFace_train'#'celeba_large_validation'
targ_dataset = 'fairface_validation'#'celeba_large_test'
query_type = 'stereotype'
att = 'gender'

results = {}
for att in ('gender','race'):
    model = 'clip-vit-large-patch14'
    with open(f'./results/ood_ref({ref_dataset})_targ({targ_dataset})_query({query_type})_att({att})_model({model}).csv', 'r') as f:
        lines = f.readlines()[1:]
    model = 'clip-vit-base-patch16'
    with open(f'./results/ood_ref({ref_dataset})_targ({targ_dataset})_query({query_type})_att({att})_model({model}).csv', 'r') as f:
        lines += f.readlines()[1:]
    latex_lines = [[round(float(num),3) for num in line.split(',')[2:]] for line in lines]
    results[att] = '\n'.join([f'{att} & {method} & ${latex_lines[i][0]} \pm {latex_lines[i][1]}$ & ${latex_lines[i][2]} \pm {latex_lines[i][3]}$& & ${latex_lines[i+len(methods)][0]} \pm {latex_lines[i+len(methods)][1]}$ & ${latex_lines[i+len(methods)][2]} \pm  {latex_lines[i+len(methods)][3]}$\\\\'
                        for i, method in enumerate(methods)])        
ref = f'''\\begin{{table}}
\centering
\\begin{{tabular}}{{cccccccc}}
& &  \multicolumn{{2}}{{c}}{{\\textbf{{ViT-L-14}}}} && \multicolumn{{2}}{{c}}{{\\textbf{{ViT-B-16}}}} \\\\ \hline
Attribute & Method & KL Div. $\downarrow$ & MaxSkew $\downarrow$ && KL Div. $\downarrow$ & MaxSkew $\downarrow$\\\\\hline
{results['race']} \hline
{results['gender']} \hline
\end{{tabular}}
\caption{{FairFace.}}
\label{{tab:methods_metrics}}
\end{{table}}'''
with open('table.tex', 'w') as f:
    f.write(ref)
