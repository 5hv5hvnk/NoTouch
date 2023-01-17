import { StyleSheet, Text, View,Pressable,TouchableOpacity,Platform } from 'react-native';
export default function LandingScreen({navigation}){
    function navToTeacher(){
        navigation.navigate("TeacherScreen")
    }
    function navToStudent(){
        navigation.navigate("StudentScreen")
    }
    return (
        
        <View style={styles.container}>
            
            
            <Pressable onPress={navToTeacher} style={Platform.OS==="ios"?styles.pressAble:styles.androidpressAble}>
                <Text style={styles.text}>Teacher</Text>
            </Pressable>
            <Pressable onPress={navToStudent} style={Platform.OS==="ios"?styles.pressAble:styles.androidpressAble}>
                <Text style={styles.text}>Student</Text>
            </Pressable>
        </View>
        
    )
}
const styles = StyleSheet.create({
    container: {
      flexDirection:"row",
      alignItems:"center",
      justifyContent:"space-around",
      marginTop:300,
    },
    androidpressAble:{
        borderColor:"#CFD2CF",
        borderRadius:7
    },
    pressAble:{
        borderWidth:2,
        borderColor:"#CFD2CF",
        borderRadius:7
    },
    text:{
        backgroundColor:'#495C83',
        color:'#ffffff',
        padding:30,
        borderRadius:100,
        fontSize:20
    }
  });