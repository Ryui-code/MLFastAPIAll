from fastapi import FastAPI
import uvicorn
from api import avocado, bank, diabetes, house, mushrooms, students, telecom, hr

app = FastAPI()

app.include_router(avocado.avocado_router)
app.include_router(bank.bank_router)
app.include_router(diabetes.diabetes_router)
app.include_router(house.house_router)
app.include_router(mushrooms.mushrooms_router)
app.include_router(students.students_router)
app.include_router(telecom.telecom_router)
app.include_router(hr.hr_router)

if __name__ == '__main__':
    uvicorn.run('main:app', reload=True)