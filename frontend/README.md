Setup Instructions

Backend

- Open a new terminal
- Create and activate a new virtual environment (optional)
- Run `pip3 install -r requirements.txt`
- Go to backend root folder
- Run `python manage.py migrate`
- Run `python manage.py createsuperuser` // Choose username and password
- Run `python manage.py runserver`

You can visit localhost:8000/admin/ for the Admin site.

