from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Housing(Base):
    __tablename__ = "housing_data"

    id = Column(Integer, primary_key=True)
    price = Column(Float)
    beds = Column(Integer)
    bath = Column(Integer)
    property_sqft = Column(Float)
    property_type = Column(String)
    borough = Column(String)
    locality = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
