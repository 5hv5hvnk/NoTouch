import * as ImagePicker from "expo-image-picker";
import { getApps, initializeApp } from "firebase/app";
import { getStorage, ref, uploadBytes, getDownloadURL,StorageReference } from "firebase/storage";
//import {View,TextInput,StyleSheet} from 'react-native'
import React, { useState } from "react";
import {
  ActivityIndicator,
  Button,
  Image,
  Share,
  StatusBar,
  StyleSheet,
  Text,
  View,
  LogBox,
  TextInput,
  Pressable,
  Platform
} from "react-native";
import * as Clipboard from "expo-clipboard";
import uuid from 'react-native-uuid'

const firebaseConfig = {
  apiKey: "AIzaSyCfQVK8oPqUgMm4pk-4oj8VOBPX0fcX3Mw",
  authDomain: "capstone-b1d1d.firebaseapp.com",
  databaseURL: "https://capstone-b1d1d-default-rtdb.firebaseio.com",
  projectId: "capstone-b1d1d",
  storageBucket: "capston-notouch",
  messagingSenderId: "141341538876",
  appId: "1:141341538876:web:83c75461f02f95e1b82139",
  measurementId: "G-3HBXCKPJYY"
};

// Editing this file with fast refresh will reinitialize the app on every refresh, let's not do that
if (!getApps().length) {
  initializeApp(firebaseConfig);
}

// Firebase sets some timeers for a long period, which will trigger some warnings. Let's turn that off for this example
LogBox.ignoreLogs([`Setting a timer for a long period`]);

export default class App extends React.Component {
  state = {
    image: null,
    uploading: false,
    rollNumber:'',
    disable:false
  };

  async componentDidMount() {
    if (Platform.OS !== "web") {
      const {
        status,
      } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== "granted") {
        alert("Sorry, we need camera roll permissions to make this work!");
      }
    }
  }
  componentWillUnmount() {
    // fix Warning: Can't perform a React state update on an unmounted component
    this.setState = (state,callback)=>{
        return;
    };
  }
  render() {
    let { image } = this.state;
    
    
    
    return (
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
        {!!image && (
          <Text
            style={{
              fontSize: 20,
              marginBottom: 20,
              textAlign: "center",
              marginHorizontal: 15,
            }}
          >
            Example: Upload ImagePicker result
          </Text>
        )}
        {!this.state.disable?<TextInput style={styles.input} placeholder='Enter your roll number' placeholderTextColor="#ffffff" keyboardType='numeric' name="txtinp" value={this.state.rollNumber} onChangeText={this._updateRoll}/>:<></>}
        <Pressable style={Platform.OS==="ios"?styles.pressAble:styles.androidpressAble} onPress={this._rollNumberSubmit} disabled={this.state.disable}>
                {!this.state.disable?<Text style={styles.text}>Submit Roll Number</Text>:<></>}
            </Pressable>
      
        <Pressable style={Platform.OS==="ios"?styles.pressAble:styles.androidpressAble} onPress={this._takePhoto}><Text style={styles.text}>Take Finger Photo</Text></Pressable>
        
        {this._maybeRenderImage()}
        {this._maybeRenderUploadingOverlay()}

        <StatusBar barStyle="default" />
      </View>
    );
  }
  _updateRoll=(txt)=>{
    //setrollNumber(txt)
    this.setState({...this.state,rollNumber:txt})
  }
  _rollNumberSubmit=()=>{
    this.setState({...this.state,disable:true})
  }
  _maybeRenderUploadingOverlay = () => {
    if (this.state.uploading) {
      return (
        <View
          style={[
            StyleSheet.absoluteFill,
            {
              backgroundColor: "rgba(0,0,0,0.4)",
              alignItems: "center",
              justifyContent: "center",
            },
          ]}
        >
          <ActivityIndicator color="#fff" animating size="large" />
        </View>
      );
    }
  };

  _maybeRenderImage = () => {
    //this.props.navigation.navigate("LandingScreen")
    let { image } = this.state;
    if (!image) {
      return;
    }

    return (
      <View
        style={{
          marginTop: 30,
          width: 250,
          borderRadius: 3,
          elevation: 2,
        }}
      >
        <View
          style={{
            borderTopRightRadius: 3,
            borderTopLeftRadius: 3,
            shadowColor: "rgba(0,0,0,1)",
            shadowOpacity: 0.2,
            shadowOffset: { width: 4, height: 4 },
            shadowRadius: 5,
            overflow: "hidden",
          }}
        >
          <Image source={{ uri: image }} style={{ width: 250, height: 250 }} />
        </View>
        <Text
          onPress={this._copyToClipboard}
          onLongPress={this._share}
          style={{ paddingVertical: 10, paddingHorizontal: 10 }}
        >
          {image}
        </Text>
      </View>
    );
  };

  _share = () => {
    Share.share({
      message: this.state.image,
      title: "Check out this photo",
      url: this.state.image,
    });
  };

  _copyToClipboard = () => {
    Clipboard.setString(this.state.image);
    alert("Copied image URL to clipboard");
  };

  _takePhoto = async () => {
    let pickerResult = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
    });

    this._handleImagePicked(pickerResult);
  };

  _pickImage = async () => {
    let pickerResult = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [4, 3],
    });

    console.log({ pickerResult });

    this._handleImagePicked(pickerResult);
  };

  _handleImagePicked = async (pickerResult) => {
    try {
      this.setState({ uploading: true });
      console.log('hello');
      console.log(pickerResult);
      if (!pickerResult.cancelled) {
        const uploadUrl = await uploadImageAsync(pickerResult.uri,this.state.rollNumber);
        this.setState({ image: uploadUrl });
        this.props.navigation.navigate("LandingScreen")
      }
    } catch (e) {
      console.log(e);
      alert("Upload failed, sorry :(");
    } finally {
      this.setState({ uploading: false });
    }
  };
}

async function uploadImageAsync(uri,rno) {
  // Why are we using XMLHttpRequest? See:
  // https://github.com/expo/expo/issues/2402#issuecomment-443726662
  const blob = await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.onload = function () {
      resolve(xhr.response);
    };
    xhr.onerror = function (e) {
      console.log(e);
      reject(new TypeError("Network request failed"));
    };
    xhr.responseType = "blob";
    xhr.open("GET", uri, true);
    xhr.send(null);
  });

  const fileRef = ref(getStorage(),`/Myimages/${rno}`);
  const result = await uploadBytes(fileRef, blob);

  // We're done with the blob, close and release it
  blob.close();
  var durl=await getDownloadURL(fileRef)
  //console.log(durl);
  return durl;
}
const styles = StyleSheet.create({
  androidpressAble:{
    borderColor:"#CFD2CF",
    borderRadius:7,
    marginVertical:50
  },
  pressAble:{
      borderWidth:2,
      borderColor:"#CFD2CF",
      borderRadius:7
  },
  text:{
      backgroundColor:'#495C83',
      color:'#ffffff',
      padding:10,
      borderRadius:100,
      fontSize:20
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

  },
  photoButton:{
    marginVertical:20,
    backgroundColor:'#7A86B6',
      borderColor:"#495C83",
  }
});