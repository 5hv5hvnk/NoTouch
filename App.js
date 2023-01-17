import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View } from 'react-native';
import {NavigationContainer} from '@react-navigation/native'
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import LandingScreen from './screens/LandingScreen';
import TeacherScreen from './screens/TeacherScreen';
import StudentScreen from './screens/StudentScreen';
import DetailsScreen from './screens/DetailsScreen';
const Stack=createNativeStackNavigator();
export default function App() {
  return (
    <>
    <StatusBar style='light'/>
    <NavigationContainer>
    <Stack.Navigator>
      <Stack.Screen 
      name="LandingScreen" 
      component={LandingScreen} 
      options={{
        title:'No-Touch',
        headerStyle:{backgroundColor:"#495C83"},
        headerTintColor:"#ffffff"
      }}/>
      <Stack.Screen 
      name="TeacherScreen" 
      component={TeacherScreen}
      options={{
        title:'Teacher Screen',
        headerStyle:{backgroundColor:"#495C83"},
        headerTintColor:"#ffffff",
        headerBackTitle:"Back"
      }}/>
      <Stack.Screen 
      name="StudentScreen" 
      component={StudentScreen}
      options={{
        title:'Student Screen',
        headerStyle:{backgroundColor:"#495C83"},
        headerTintColor:"#ffffff",
        headerBackTitle:"Back"
      }}
      />
      <Stack.Screen 
      name="DetailsScreen" 
      component={DetailsScreen}
      options={{
        title:'Details Screen',
        headerStyle:{backgroundColor:"#495C83"},
        headerTintColor:"#ffffff",
        headerBackTitle:"Back"
      }}
      />
    </Stack.Navigator>
    </NavigationContainer>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
