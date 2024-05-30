import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict
import pandas as pd
import gurobipy
from tdvrp import TDVRP


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

locations_data: List[Dict[str, float]] = []
df = None

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    global df
    try:
        df = pd.read_csv(file.file)
        global locations_data
        locations_data = df.to_dict(orient='records')
        print(locations_data)

        return JSONResponse(content={"status": "success"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/locations", response_model=List[Dict[str, float]])
async def get_locations():
    return locations_data

@app.post("/tdvrp")
async def tdvrp(nDrones: str = Form(...), 
                nTrucks: str = Form(...), 
                drone_speed: str = Form(...), 
                truck_speed: str = Form(...),
                drone_capacity: str = Form(...),
                truck_capacity: str = Form(...),
                drone_autonomy: str = Form(...),
                truck_threshold: str = Form(...)):
    
    tdvrp = TDVRP()
    df['demand'] = df['demand'].astype(int)
    demands = df['demand'].tolist()
    matrix_distance_truck = tdvrp.distance_matrix_truck(df)
    matrix_time_truck = tdvrp.time_matrix(matrix_distance_truck, float(truck_speed))
    matrix_distance_drone = tdvrp.calculate_drone_distance_matrix(df)
    matrix_time_drone = tdvrp.time_matrix(matrix_distance_drone, float(drone_speed))
    n = len(locations_data)
    customers_list = [x for x in range(1, n)]
    nodes_list = [0] + customers_list
    list_vars = []
    dic_res = dict()
    resultsTDVRP = tdvrp.TDVRP(dem=demands, 
                               time_truck=matrix_time_truck, 
                               customers=customers_list, 
                               nodes=nodes_list, 
                               nT=int(nTrucks),
                               truck_capacity=int(truck_capacity),
                               nD=int(nDrones),
                               drone_capacity=int(drone_capacity),
                               drone_endurance=int(drone_autonomy),
                               time_drone=matrix_time_drone)
    
    model = resultsTDVRP[0]
    xt = resultsTDVRP[1]
    xd = resultsTDVRP[2]
    list_vars.append(xt)
    list_vars.append(xd)
    dic_res[model] = list_vars

    numIter = 1
    for mdl in dic_res:
        vars = []
        routesT = []
        routesD= []
        print("===================model==========:", mdl)
        name = mdl.ModelName
        vars = dic_res[mdl]
        xt = vars[0]
        xd = vars[1]
        runTimeTot = 0
        for i in range(1, numIter+1):
            res = tdvrp.solving_model(mdl)
            status = res[0]
            runTime = res[1]
            TravTime = res[2]
            #TravTime = 2*sum(time_truck[0]) + 2*sum(time_drone[0])- savCost
            runTimeTot = runTimeTot + runTime
            runTimeAvg = runTimeTot/numIter
            if TravTime ==0:
                break
            print("run time average TDVRP: ", runTimeAvg)
            print("Travel time ", TravTime)
            #print("Travel Time ", TravTime)
            mdl.write("modelTDVRP2F.lp")
            if status == 2:
                mdl.write("modelTDVRP2F.lp")
                all_vars = mdl.getVars()
                values = mdl.getAttr("X", all_vars)
                names = mdl.getAttr("VarName", all_vars)
#                 for name, val in zip(names, values):
#                     print(f"{name} = {val}")
#                         plot_solution_WV(nodes,mdl, df, xt, xd)
                vals_t = mdl.getAttr('X', xt)
                arcs_t = [(i,j) for i, j in vals_t.keys() if vals_t[i, j] > 0.99]
                routesT = tdvrp.extract_routes(arcs_t)
                vals_d = mdl.getAttr('X', xd)
                arcs_d = [(i,j) for i, j in vals_d.keys() if vals_d[i, j] > 0.99]
                print("**************************arcsT********************************", arcs_t)
                print("**************arcD ",arcs_d)
                routesT = tdvrp.extract_routes(arcs_t)
                routesD = tdvrp.extract_routes(arcs_d)
                print("**************************routesT********************************", routesT)
                print("*********************routesD", routesD)
            else:
                try:
                    mdl.computeIIS()
                    mdl.write('iismodelTDVRP2F.ilp')
                    print('\nThe following constraints and variables are in the IIS:')
                    for c in mdl.getConstrs():
                        if c.IISConstr: print(f'\t{c.constrname}: {mdl.getRow(c)} {c.Sense} {c.RHS}')

                    for v in mdl.getVars():
                        if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
                        if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
                except gurobipy.GurobiError as e:
                    if "Cannot compute IIS on a feasible model" in str(e):
                        print("Model is feasible, no IIS found.")
                    else:
                        raise e
    
    return routesT, routesD

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)