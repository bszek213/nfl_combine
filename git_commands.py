#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:36:21 2021

@author: bszekely

Automate Github bash commands only works for Linux 
"""
import subprocess
import yaml

def import_yaml():
    message1 = input("\nYAML Directory\n")
    with open(message1, "r") as stream:
        documents = yaml.safe_load(stream)
        return documents['Token'],documents['repo'],documents['branch'],documents['branch_exist']
      
def run(*args):
    return subprocess.check_call(['git'] + list(args))

def initialize():
    message = input("do you want to initialize an empty git repo? yes/no")
    if message.lower() == "yes":
        run("init")
    else:
        print("git is already initialized, proceed")

def add():
    message = input("\nDo you want to git add all files or specific? all/choose \n")
    if message == "all":
        run("add","--all")
    elif message == "choose":
        message2 = input("\nwhat files would you like to input\n")
        run("add", f'{message2}')
def commit():
    message = input("\ncommit message:\n")
    run('commit', "-m", message)

def push():
    message = input("\nDo you want to push these changes? y or n\n")
    if message == "no":
        print("exiting push command")
    else:
        Token,repo,branch,branch_exist = import_yaml()
        print(branch_exist)
        if branch_exist == False:
            run("checkout","-b", branch)
            print("MAKE THIS BRANCH YA TWIT")
        combine_str = "https://" + Token + "@github.com/" + repo
        run("push","--set-upstream",combine_str,branch)
   
if __name__ == '__main__':
    while True:
        print("\nWhat git commands do you want to execute?")
        print("init - initialize an empty git repo")
        print("add - add specific or all files to your git repo")
        print("commit - commit these changes with an input message")
        print("push - push these changes to your repo from yaml")
        print("end - exit script")
        command_prompt = input("\ninput what git command you want\n")
        command_prompt = command_prompt.lower()
        if command_prompt == "add":
            add()
        elif command_prompt == "init":
            initialize()
        elif command_prompt == "commit":
            commit()
        elif command_prompt == "push":
            push()
        elif command_prompt == "end":
            break
