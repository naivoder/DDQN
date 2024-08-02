# Double Deep Q Network

## Overview

🚧 🛠️👷‍♀️ 🛑 Under construction...

## Setup

### Required Dependencies

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Running the Algorithm

You can run the algorithm on any supported Gymnasium environment. For example:

```bash
python main.py --env 'LunarLander-v2'
```

---


<table>
    <tr>
        <td>
            <p><b>MsPacman</b></p>
            <img src="environments/MsPacmanNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>AirRaid</b></p>
            <img src="environments/AirRaidNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Alien</b></p>
            <img src="environments/AlienNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/MsPacmanNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AirRaidNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AlienNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td> 
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Amidar</b></p>
            <img src="environments/AmidarNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Assault</b></p>
            <img src="environments/AssaultNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Asterix</b></p>
            <img src="environments/AsterixNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/AmidarNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AssaultNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AsterixNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Asteroids</b></p>
            <img src="environments/AsteroidsNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Atlantis</b></p>
            <img src="environments/AtlantisNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>BankHeist</b></p>
            <img src="environments/BankHeistNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/AsteroidsNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AtlantisNoFrameskip-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BankHeistNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>BattleZone</b></p>
            <img src="environments/BattleZoneNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>BeamRider</b></p>
            <img src="environments/BeamRiderNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Berzerk</b></p>
            <img src="environments/BerzerkNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/BattleZoneNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BeamRiderNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BerzerkNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Bowling</b></p>
            <img src="environments/BowlingNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Boxing</b></p>
            <img src="environments/BoxingNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Breakout</b></p>
            <img src="environments/BreakoutNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/BowlingNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BoxingNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BreakoutNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Carnival</b></p>
            <img src="environments/CarnivalNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Centipede</b></p>
            <img src="environments/CentipedeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>ChopperCommand</b></p>
            <img src="environments/ChopperCommandNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/CarnivalNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/CentipedeNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/ChopperCommandNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>CrazyClimber</b></p>
            <img src="environments/CrazyClimberNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Defender</b></p>
            <img src="environments/DefenderNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>DemonAttack</b></p>
            <img src="environments/DemonAttackNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/CrazyClimberNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/DefenderNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/DemonAttackNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>DoubleDunk</b></p>
            <img src="environments/DoubleDunkNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>ElevatorAction</b></p>
            <img src="environments/ElevatorActionNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Enduro</b></p>
            <img src="environments/EnduroNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/DoubleDunkNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/ElevatorActionNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/EnduroNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>FishingDerby</b></p>
            <img src="environments/FishingDerbyNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Freeway</b></p>
            <img src="environments/FreewayNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Frostbite</b></p>
            <img src="environments/FrostbiteNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/FishingDerbyNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/FreewayNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/FrostbiteNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Gopher</b></p>
            <img src="environments/GopherNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Gravitar</b></p>
            <img src="environments/GravitarNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Hero</b></p>
            <img src="environments/HeroNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/GopherNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/GravitarNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/HeroNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>IceHockey</b></p>
            <img src="environments/IceHockeyNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>JamesBond</b></p>
            <img src="environments/JamesbondNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>JourneyEscape</b></p>
            <img src="environments/JourneyEscapeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/IceHockeyNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/JamesbondNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/JourneyEscapeNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Kangaroo</b></p>
            <img src="environments/KangarooNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Krull</b></p>
            <img src="environments/KrullNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>KungFuMaster</b></p>
            <img src="environments/KungFuMasterNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/KangarooNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/KrullNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/KungFuMasterNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>MontezumaRevenge</b></p>
            <img src="environments/MontezumaRevengeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <!--<td>
            <p><b>Krull</b></p>
            <img src="environments/KrullNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>KungFuMaster</b></p>
            <img src="environments/KungFuMasterNoFrameskip-v4.gif" width="250" height="250"/>
        </td>-->
    </tr>
    <tr>
        <td>
            <img src="metrics/MontezumaRevengeNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <!--<td>
            <img src="metrics/KrullNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/KungFuMasterNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>-->
    </tr>
</table>

## Acknowledgements

Special thanks to Phil Tabor, an excellent teacher! I highly recommend his [Youtube channel](https://www.youtube.com/machinelearningwithphil).
