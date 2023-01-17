import { StyleSheet, Text, View,Pressable,Button,Platform } from 'react-native';
import { useState } from 'react';
import LandingScreen from './LandingScreen';
//import axios from 'axios';
const TeacherScreen=({navigation})=>{
    const [genOTP,setgenOTP]=useState(0)
    const generateOTP=()=>{
        const random = Math.floor(Math.random() * 9000 + 1000);
        fetch('https://capstone-b1d1d-default-rtdb.firebaseio.com/OTP.json',{
                    method:'DELETE'
                  })
        setgenOTP(random);
    }
    const submitOTP=()=>{

       
       fetch('https://capstone-b1d1d-default-rtdb.firebaseio.com/OTP.json',{
                method:'POST',
                headers:{
                    'Content-Type':'application/json'
                },
                body : JSON.stringify({
                   genOTP
                })
            }).then(()=>{
                //console.log('done')
                navigation.navigate('LandingScreen')
                setTimeout(()=>{
                  fetch('https://capstone-b1d1d-default-rtdb.firebaseio.com/OTP.json',{
                    method:'DELETE'
                  })
                },600000)
            })
        
        
    }
    return (
        <View style={styles.container}>
            {genOTP===0?<Pressable style={Platform.OS==="ios"?styles.pressAble:styles.androidpressAble} onPress={generateOTP} ><Text style={styles.text}>Generate 4 digit OTP</Text></Pressable>:<></>}
            
            
            {genOTP===0?<></>:<Pressable style={Platform.OS==="ios"?styles.pressAble:styles.androidpressAble}><Text style={styles.text}>OTP generated {genOTP}</Text></Pressable>}
            {genOTP===0?<></>:<Pressable style={Platform.OS==="ios"?styles.submitButton:styles.androidSubmitButton} onPress={submitOTP}>
                   <Text style={styles.text}>Submit</Text>
            </Pressable>}
            </View>
    )
}
const styles = StyleSheet.create({
    container: {
        flexDirection:'column',
        backgroundColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
      },
      androidpressAble:{
        borderColor:"#CFD2CF",
        borderRadius:7,
        marginVertical:50

      },
      pressAble:{
          borderWidth:2,
          borderColor:"#CFD2CF",
          borderRadius:7,
          marginVertical:100
      },
      androidSubmitButton:{
        borderColor:"#CFD2CF",
        borderRadius:7,
        marginBottom:20
      },
      submitButton:{
        borderWidth:2,
        borderColor:"#CFD2CF",
        borderRadius:7,
        marginBottom:20
      },
      text:{
          backgroundColor:'#495C83',
          color:'#ffffff',
          padding:10,
          borderRadius:100,
          fontSize:20,
      },
      input:{
          backgroundColor:'#7A86B6',
          borderColor:"#495C83",
          borderWidth:4,
          borderRadius:20,
          padding:20,
          marginVertical:20,
          color:'#ffffff',
          width:350
  
      }
  });
  export default TeacherScreen;