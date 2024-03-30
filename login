/* login */
:root{
    --lightblue: rgb(219, 181, 151);
    --darkblue:rgb(0, 51, 65);
    --white:#fff;
    --main-color: #0d6d7c;
    --bg-color: #fff;
    --text-color: #0f0c27;
    --hover:rgb(0, 0, 0);
    --hover2:hsl(180, 6%, 46%);;
    --form:hsla(204, 100%, 62%, 0.1);
    --big-font: 3.2rem;
    --medium-font: 1.8rem;
    --p-font: 1rem;
    --blue:rgb(0, 65, 102);
    --bg:#f7f3ef;
    --yellow:#fdc755
   
}
body{
   /* admin login css */
    background: url(/static/photos/bg2.png);
    background-position: left;
    background-size: cover;
    background-repeat: no-repeat;
    min-height: 100vh; 

 /* background-size: 50% 100%; */
 /* background: var(--white-3); */

 

 
   
}
/* body #login{
    background: url(/static/photos/bglogin.png);
    background-position: left;
    background-size: cover;
    background-repeat: no-repeat;
   
} */
.login,.wrapper{
    padding-top: 60px;
    
   
    display: flex;
    align-items: flex-end;
    justify-content: flex-end;
}
.button-login{
    padding: 0.51rem;
    padding-right: 1rem;
  border: 3px solid #468a89;
  background: transparent;
  border-radius: 2rem;
  
}
.wrapper{
    position: fixed;
    /* top: 0.5rem; */
    /* right: 0;  */
    top: 50%;
    left: 75%;
    transform: translate(-50%,-50%);
    width: 450px;
    height:600px;
    background: transparent;
   
    backdrop-filter: blur(30px);
    /* box-shadow: -1px 0 10px rgba(72, 255, 0, 0.2);
    border: 2px solid rgba(0, 255, 8, 0.1); */
   
    /* z its goes over the header if 1000 under the header */
    /* z-index: 101; */
    display: flex;
    align-items: center;
    padding: 50px 60px 70px;
    /* opacity: 0; */
    justify-content: center;
    margin-top: 2rem;
    box-shadow: 28px 28px 10px rgba(82, 35, 0, 0.2);
    /* border: 5px solid #a6e4ff; */
   
}
 .wrapper.active-popup{
    opacity: 0;
    
    pointer-events: inherit;
} 
 .icon-close{
    position: fixed;
    padding: 1%;
    /* or =>width: 60px;
    height: 60px; */
    background: var(--main-color);
    /* for it to be on top right */
    top: 0;
    right: 0;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    border-bottom-left-radius: 10px;
}
.icon-close i{
    font-size: 32px;
    color: var(--dark);
}
.wrapper .logreg-box{
    width: 100%;
    /* background: var(--pink); */
}
.wrapper .form-box.login{
    display: block;
}
/* .wrapper.active .form-box.login{
    display: none;
} */
.wrapper .form-box.register{
    display: block;
}
/* .wrapper.active .form-box.register{
    display: block;
} */
.logreg-box .logreg-title{
    text-align: center;
    margin-bottom: 40px;
    color: var(--dark);
}
.logreg-title h2{
    font-size: 32px;
    color: var(--dark);
}
.logreg-title p{
    padding-top: 0.5rem;
    font-size: 20px;
    font-weight: 500;
    color: var(--yellow);
}
.logreg-box .input-box{
    border: 3px solid var(--bg);
    position: relative;
    width: 100%;
    height: 50px;
    /* background: red; */
    margin: 30px 0;
    transition: border-color var(--main-color);
    color: var(--dark);
}
.logreg-box .input-box:hover{
    border: 3px solid var(--dark);

}
.input-box input{
    width: 100%;
    height: 100%;
    background: transparent;
    border: none;
    outline: none;
    font-size: 16px;
    font-weight: 500;
    color: var(--dark);
    
    padding-right: 25px;

}
.input-box label{
    position: absolute;
    left: 0;
    font-size: 16px;
    color: var(--dark);
    font-weight: 500;
    top: 50%;
    transform: translateY(-90%);
    pointer-events: none;
    transition: .5s;

}
.input-box input:focus~label,
.input-box input:valid~label{
    top: -5px;
    /* to move email and password */
}
.input-box .icon{
      position: absolute;  
     top: 50%; 
 right: 0;  
     transform: translateY(-50%); 
     color: var(--dark); 
     font-size: 19px; 

}
.logreg-box .remember-forgot{
    font-size: 14.5px;
    color: var(--dark);
    font-weight: 500;
    margin: -15px 0 15px;
    display: flex;
    justify-content: space-between;
}
.remember-forgot label input{
    accent-color: var(--dark);;
    margin-right: 3px;
}
.remember-forgot a{
    color: var(--dark);
    text-decoration: none;
    font-size: 16px;
}
.remember-forgot a:hover{
    text-decoration: underline;
}
.logreg-box .btn{
     width: 100%;
    /* height: 20px;  */
    padding: 2.5%;
    background: var(--yellow) ;
    
    border: none;
    outline: none;
    border-radius: 40px;
    box-shadow: 0 2px 5px rgba(0,0,0,.2);
    cursor: pointer;
    font-size: 20px;
    color: var(--hover);
    font-weight: 600;

}
.logreg-box .logreg-link p{
    font-size: 14.5px;
    color: var(--dark);
    text-align: center;
    font-weight: 500;
    margin-top: 25px;
}
.logreg-box .logreg-link p a{
    color: var(--dark);
    text-decoration: none;
    font-weight: 600; 
    font-size: 16px;
}
.logreg-box .logreg-link p a:hover{
    text-decoration: underline;
}
@media (max-width:700px){
    body{
        background-size: cover;
        height: 100%;
        background-position: center;
    }
    body{
        /* admin login css */
         /* background: url(images/bg.jpg); */
         /* background-position: center; */
          background-size: cover;
      /* background-repeat: no-repeat; */
      height: 100%; 
      /* background-image: url('images/hotel2.jpg'); */
      background-repeat: no-repeat;
      /* background-size: 50% 100%; */
      /* background: var(--white-3); */
   
     }
     .wrapper{
        left: 50%;
        backdrop-filter: blur(50px);
        box-shadow: 28px 28px 10px rgba(255, 244, 235, 0.2);
        border: 10px solid var(--white-3);  
    }
    .logreg-box .logreg-title{
      
        color: var(--dark);
    }
    .logreg-title h2{
        text-shadow: 3px 3px var(--dark);
        color: var(--white-2);
    }
    .logreg-title p{
       
        color: var(--white-2);
    } 
    .input-box input{
     
        color: var(--white-3);  
    
    }
    .input-box label{
        
        color: var(--white-2);
    }
   
    .input-box .icon{
          position: absolute;  
         top: 50%; 
     right: 0;  
         transform: translateY(-50%); 
         color: var(--orange); 
         font-size: 19px; 
    
    }
    .logreg-box .remember-forgot{
      
        color: var(--white-2);
       
    }
    
    .remember-forgot a{
        color: var(--white-3);
       
    }
   
    .logreg-box .btn{
       
        color: var(--white-3);
     
    }
    .logreg-box .logreg-link p{
     
        color: var(--white-2);
      
    }
    .logreg-box .logreg-link p a{
        color: var(--white-3);
        
    }

}
