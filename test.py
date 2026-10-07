import pysubgroup as ps
from pysubgroup.datasets import get_titanic_data
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


data = get_titanic_data()
target = ps.BinaryTarget("Survived", True)
searchspace = ps.create_selectors(data, ignore=["Survived"])

def make_task():
    return ps.SubgroupDiscoveryTask(
        data, target, searchspace, result_set_size=5, depth=2, qf=ps.WRAccQF())

result = ps.DFS().execute(make_task())
print(result.to_dataframe())

result_gp = ps.GpGrowth().execute(make_task())
print(result_gp.to_dataframe())