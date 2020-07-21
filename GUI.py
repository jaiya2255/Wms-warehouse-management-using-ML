from tkinter import *
import sys
import csv
from tempfile import NamedTemporaryFile
import shutil
#PRODUCT MENU
def clickedbtn1():
    medicine_menu_window = Tk()
    medicine_menu_window.geometry('500x400')
    medicine_menu_window.title("Inventory Management")
    lbl = Label(medicine_menu_window, text="GAIL.co")
    lbl.grid(column=0, row=0)
    lbl2 = Label(medicine_menu_window, text="What would you like to do!", bg="blue",width="75", height="2")
    lbl2.grid(column=0, row=1)
    btn1 = Button(medicine_menu_window, text="ADD NEW PRDT",fg="green", command=mclickedbtn1)
    btn1.grid(column=0, row=2)
    btn2 = Button(medicine_menu_window, text="SEARCH PRDT",fg="green", command=mclickedbtn2)
    btn2.grid(column=0, row=3)
    btn3 = Button(medicine_menu_window, text="UPDATE PRDT",fg="green", command=mclickedbtn3)
    btn3.grid(column=0, row=4)
    btn4 = Button(medicine_menu_window, text="PRDT TO BE PURCHASED",fg="green", command=mclickedbtn4)
    btn4.grid(column=0, row=5)
    btn4 = Button(medicine_menu_window, text="RETURN TO MAIN MENU",fg="green", command=mclickedbtn4)
    btn4.grid(column=0, row=6)
    medicine_menu_window.mainloop()
#EMPLOYEE MENU
def clickedbtn2():
    c_menu_window = Tk()
    c_menu_window.geometry('500x400')
    c_menu_window.title("Inventory Management")
    lbl = Label(c_menu_window, text="EMPLOYEE MENU")
    lbl.grid(column=0, row=0)
    lbl2 = Label(c_menu_window, text="What would you like to do!",bg="blue", width="75", height="2")
    lbl2.grid(column=0, row=1)
    btn1 = Button(c_menu_window, text=" NEW_EMPLOYEE",fg="green", command=mclickedbtn1)
    btn1.grid(column=0, row=2)
    btn2 = Button(c_menu_window, text=" SEARCH_EMPLOYEE",fg="green", command=mclickedbtn2)
    btn2.grid(column=0, row=3)
    btn3 = Button(c_menu_window, text=" UPDATE EMPLOYEE INFORMATION",fg="green", command=mclickedbtn3)
    btn3.grid(column=0, row=4)
    btn4 = Button(c_menu_window, text="RETURN TO MAIN MENU",fg="green", command=mclickedbtn4)
    btn4.grid(column=0, row=5)
    c_menu_window.mainloop()
    
#SUPPLIER MENU
def clickedbtn3():
    s_menu_window = Tk()
    s_menu_window.geometry('500x400')
    s_menu_window.title("Inventory Management")
    lbl = Label(s_menu_window, text="SUPPLIER MENU")
    lbl.grid(column=0, row=0)
    lbl2 = Label(s_menu_window, text="What would you like to do!", bg="blue", width="75", height="2")
    lbl2.grid(column=0, row=1)
    btn1 = Button(s_menu_window, text="TENDER CALL REPORT ",fg="green", command=mclickedbtn1)
    btn1.grid(column=0, row=2)
    btn2 = Button(s_menu_window, text="SUPPLIER INFO ",fg="green", command=mclickedbtn2)
    btn2.grid(column=0, row=3)
    btn3 = Button(s_menu_window, text="UPDATE TENDER BID ANALYSIS INFO",fg="green", command=mclickedbtn3)
    btn3.grid(column=0, row=4)
    btn3 = Button(s_menu_window, text="PERFORMANCE BANK GUARANTEE REPRORT",fg="green", command=mclickedbtn3)
    btn3.grid(column=0, row=4)
    btn4 = Button(s_menu_window, text="RETURN TO MAIN MENU",fg="green", command=mclickedbtn4)
    btn4.grid(column=0, row=5)
    s_menu_window.mainloop()
#REPORT MENU
def clickedbtn4():
    r_menu_window = Tk()
    r_menu_window.geometry('500x400')
    r_menu_window.title("Inventory Management")
    lbl = Label(r_menu_window, text="REPORT MENU")
    lbl.grid(column=0, row=0)
    lbl2 = Label(r_menu_window, text="What would you like to do!",bg="blue", width="75", height="2")
    lbl2.grid(column=0, row=1)
    btn1 = Button(r_menu_window, text="OVERSTOCK PRODUCTS",fg="green", command=mclickedbtn1)
    btn1.grid(column=0, row=2)
    btn2 = Button(r_menu_window, text="BACKORDER PRODUCTS ",fg="green", command=mclickedbtn2)
    btn2.grid(column=0, row=3)
    btn3 = Button(r_menu_window, text="LIFESPAN OF PRODUCT ",fg="green", command=mclickedbtn3)
    btn3.grid(column=0, row=5)
    btn4 = Button(r_menu_window, text="REPRORT OF DIFFERENT S.LOC",fg="green", command=mclickedbtn3)
    btn4.grid(column=0, row=6)
    btn5 = Button(r_menu_window, text="EXPIRED PRODUCT",fg="green", command=mclickedbtn3)
    btn5.grid(column=0, row=7)
    btn6 = Button(r_menu_window, text="SCRAPE REPORT",fg="green", command=mclickedbtn3)
    btn6.grid(column=0, row=8)
    btn7 = Button(r_menu_window, text="SCRAPE OVERSTOCK",fg="green", command=mclickedbtn3)
    btn7.grid(column=0, row=9)
    btn8 = Button(r_menu_window, text="RETURN TO MAIN MENU",fg="green", command=mclickedbtn4)
    btn8.grid(column=0, row=10)
    r_menu_window.mainloop()
#INVOICING MENU
def clickedbtn5():
    r_menu_window = Tk()
    r_menu_window.geometry('500x400')
    r_menu_window.title("Inventory Management")
    lbl = Label(r_menu_window, text="INVOICING MENU")
    lbl.grid(column=0, row=0)
    lbl2 = Label(r_menu_window, text="What would you like to do!", bg="blue", width="75", height="2")
    lbl2.grid(column=0, row=1)
    btn1 = Button(r_menu_window, text="SUPPLIER INVOICE",fg="green", command=mclickedbtn1)
    btn1.grid(column=0, row=2)
    btn2 = Button(r_menu_window, text="CUSTOMER INVOICE",fg="green", command=mclickedbtn2)
    btn2.grid(column=0, row=3)
    btn3 = Button(r_menu_window, text="ADD VOUCHER",fg="green", command=mclickedbtn1)
    btn3.grid(column=0, row=4)
    btn4 = Button(r_menu_window, text="VIEW SIV VOUCHERS",fg="green", command=mclickedbtn2)
    btn4.grid(column=0, row=5)
    btn5 = Button(r_menu_window, text="VIEW GRV VOUCHERS",fg="green", command=mclickedbtn2)
    btn5.grid(column=0, row=6)
    btn6 = Button(r_menu_window, text="RETURN TO MAIN MENU",fg="red", command=mclickedbtn4)
    btn6.grid(column=0, row=7)
    r_menu_window.mainloop()

#Main Menu
window = Tk()
window.geometry('500x400')
window.title("Inventory Management")
lbl = Label(window, text="GAIL.CO")
lbl.grid(column=0, row=0)
lbl2 = Label(window, text="What would you like to do!", bg="blue", width="75", height="2")
lbl2.grid(column=0, row=1)
btn1 = Button(window, text="PRODUCT MENU",fg="BLUE", command=clickedbtn1)
btn1.grid(column=0, row=2)
btn2 = Button(window, text="EMPLOYEE MENU",fg="blue", command=clickedbtn2)
btn2.grid(column=0, row=3)
btn3 = Button(window, text="SUPPLIER MENU",fg="blue", command=clickedbtn3)
btn3.grid(column=0, row=4)
btn4 = Button(window, text="REPORT MENU",fg="blue", command=clickedbtn4)
btn4.grid(column=0, row=5)
btn5 = Button(window, text="INVOICING MENU",fg="blue", command=clickedbtn5)
btn5.grid(column=0, row=6)
window.mainloop()
#---------------------------------------------------------------------------------------EMPLOYEE DATA---------------------------------------------------------------------------------------#
import csv
from tempfile import NamedTemporaryFile
import shutil

def customer_id_generator():
    with open('cus_men.csv','r') as csvfile:
        reader=csv.DictReader(csvfile)
        i=1
        for row in reader:
            if int(row['customer_id'])==i:
                i=i+1            
    return i

def new_customer():
    with open('cus_men.csv','a+') as csvfile:
        names=['customer_name','customer_id','customer_phone','customer_address']
        writer=csv.DictWriter(csvfile,fieldnames=names)
        writer.writeheader()
        customer_name=input('Enter the name of the customer : ')
        customer_id=customer_id_generator()
        print('Unique customer ID generated : ',customer_id)
        customer_phone=input('Enter the phone number of the customer : ')
        customer_address=input('Enter the address : ')
        writer.writerow({'customer_name':customer_name,'customer_id':customer_id,'customer_phone':customer_phone,"customer_address":customer_address})

def search_customer():    
    with open('cus_men.csv','r') as csvfile:
        name=input('Enter the name of customer:\n')
        reader=csv.DictReader(csvfile)
        for row in reader:
            if row['customer_name']==name:
                print("------------------------------------------")
                print(" Name : ",row['customer_name'],'\n',"ID : ",row['customer_id'],'\n',"Phone : ",row['customer_phone'],'\n',"Address : ",row['customer_address'])
                print("------------------------------------------")
def update_customer_info():
    tempfile = NamedTemporaryFile(mode='w', delete=False)
    names=['customer_name','customer_id','customer_phone','customer_address']
    with open('cus_men.csv', 'r') as csvfile, tempfile:
        reader = csv.DictReader(csvfile)
        writer = csv.DictWriter(tempfile, fieldnames=names)
        writer.writeheader()
        idno =input('Enter the id of the customer you want to modify!\n')
        for row in reader:
            if row['customer_id'] == idno:
                print('---------------------------------------------')
                print("|Enter 1 to change name                     |")
                print('---------------------------------------------')
                print('|Enter 2 to change phone number             |')
                print('---------------------------------------------')
                print('|Enter 3 to change address                  |')
                print('---------------------------------------------') 
                choice=int(input("Enter Your Choice!\n"))

                if(choice==1):
                    row['customer_name']=input("Enter the new name : ")

                elif(choice==2):
                    row['customer_phone']=input("Enter the new phone number : ")

                elif(choice==3):

                    row['customer_address']=input("Enter the new address : ")

            row = {'customer_name':row['customer_name'],'customer_id':row['customer_id'],'customer_phone':row['customer_phone'],"customer_address":row['customer_address']}
            writer.writerow(row)

    shutil.move(tempfile.name, 'cus_men.csv')
