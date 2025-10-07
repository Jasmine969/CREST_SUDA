from socket import gethostname

host2path = {
    'LAPTOP-1QA0JPIO': 'F:/intestine_results',
    'DESKTOP-EHK58OI': 'F:/EntericNervousSystem/my_work/results',
    'gpu-server': '/data/zhuhong_codes/EntericNervousSystem/my_work/results',
    'mu02': '/home/zhuhong/EntericNervousSystem/my_work/results',
    'cu14': '/home/zhuhong/EntericNervousSystem/my_work/results',
    'cu15': '/home/zhuhong/EntericNervousSystem/my_work/results',
    'cu16': '/home/zhuhong/EntericNervousSystem/my_work/results',
    'cu17': '/home/zhuhong/EntericNervousSystem/my_work/results'
}
hostname = gethostname()
if hostname in host2path:
    RES_PATH: str = host2path[gethostname()]
else:
    RES_PATH = '../my_work/results'
