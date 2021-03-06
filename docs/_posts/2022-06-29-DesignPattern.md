---
title:  "Design Pattern"
date:   2022-06-29 12:00:04 -0700
categories: 
 - Languages
toc: true
toc_sticky: true
---

### Overview

- 在软件工程中,设计模式(design pattern)是对软件设计中普遍存在(反复出现)的各种问题,所提出的解决方案。
- Definition: a reusable solution to a commonly occurring problem within a given context. It’s a template or description for how to solve a problem.

### Category

- 创建型模式(对象的创建)：简单工厂模式(使用静态方法,根据参数返回不同产品),工厂方法模式(对每一种产品提供一个工厂类,支持增加新产品),抽象工厂模式(应对产品族概念,支持增加新产品族),创建者模式,原型模式,单例模式
- 结构型模式(对象的结构)：外观模式,适配器模式,代理模式,装饰者模式,桥模式,组合模式,享元模式
- 行为型模式(对象的行为)：模板方法模式,观察者模式,状态模式,策略模式,职责链模式,命令模式,访问者模式,调停者模式,备忘录模式,迭代器模式,解释器模式
- 代理模式：当客户端代码需要调用某个对象时,不关心是否准确地得到了某个对象。如Hibernate的延时加载,只在真正调用的时候才去加载。
- 命令模式：客户端不知道具体的操作,而由Command来决定,客户端只负责调用Command的方法。
- 策略模式：用来封装一系列的算法,通常被封装在Context类中,客户端可以自由选择算法。
- 门面模式(Facade)：将一组复杂的类包装到一个简单的外部接口中。随着系统的发展,程序的流程会越来越复杂,门面模式可以提供一个简化的接口,从而简化这些类的复杂性。
- 桥接模式：由于实际的需要,某个类具有两个以上的维度变化,如果只是使用继承将无法实现这种需要,或者使得设计变得相当臃肿。而桥接模式的做法是把变化部分抽象出来,使变化部分与主类分离开来,从而将多个的变化彻底分离。最后提供一个管理类来组合不同维度上的变化,通过这种组合来满足业务的需要
- 观察者模式：
    1. 主题：主题是一个接口,该接口规定了具体主题需要实现的方法,比如添加、删除观察者以及通知观察者更新数据的方法。
    2. 观察者：观察者也是一个接口,该接口规定了具体观察者用来更新数据的方法。
    3. 具体主题：具体主题是一个实现主题接口的类,该类包含了会经常发生变化的数据。而且还有一个集合,该集合存放的是观察者的引用。
    4. 具体观察者：具体观察者是实现了观察者接口的一个类。具体观察者包含有可以存放具体主题引用的主题接口变量,以便具体观察者让具体主

### **Adapter适配器模式**

- 用来把一个接口转化成另一个接口。使得原本由于接口不兼容而不能一起工作的那些类可以在一起工作。
- 例子
    - java.util.Arrays#asList()
    - java.io.InputStreamReader(InputStream)
    - java.io.OutputStreamWriter(OutputStream)

```java
public interface RowingBoat {
  void row();
}

public class FishingBoat {
  public void sail() {
    LOGGER.info("The fishing boat is sailing");
  }
}

public class FishingBoatAdapter implements RowingBoat {

  private FishingBoat boat;

  public FishingBoatAdapter() {
    boat = new FishingBoat();
  }

  @Override
  public void row() {
    boat.sail();
  }
}
```

### **Decorator装饰者模式 (a.k.a wrapper)**

- 动态的给一个对象添加额外的功能,这也是子类的一种替代方式。可以看到,在创建一个类型的时候,同时也传入同一类型的对象。
- 例子
    - java.io.BufferedInputStream(InputStream)
    - java.io.DataInputStream(InputStream)
    - java.io.BufferedOutputStream(OutputStream)

```java
public interface Troll {
  void attack();
  int getAttackPower();
  void fleeBattle();
}

public class ClubbedTroll implements Troll {
  private Troll decorated;

  public ClubbedTroll(Troll decorated) {
    this.decorated = decorated;
  }

  @Override
  public void attack() {
    decorated.attack();
    LOGGER.info("The troll swings at you with a club!");
  }

  @Override
  public int getAttackPower() {
    return decorated.getAttackPower() + 10;
  }

  @Override
  public void fleeBattle() {
    decorated.fleeBattle();
  }
}
```

### **Facade门面模式/外观模式**

- 给一组组件,接口,抽象,或者子系统提供一个简单的接口。
- 例子
    - java.lang.Class
    - javax.faces.webapp.FacesServlet

```java
public class DwarvenGoldmineFacade {

  private final List<DwarvenMineWorker> workers;

  public DwarvenGoldmineFacade() {
    workers = new ArrayList<>();
    workers.add(new DwarvenGoldDigger());
    workers.add(new DwarvenCartOperator());
    workers.add(new DwarvenTunnelDigger());
  }

  public void startNewDay() {
    makeActions(workers, DwarvenMineWorker.Action.WAKE_UP, DwarvenMineWorker.Action.GO_TO_MINE);
  }

  public void digOutGold() {
    makeActions(workers, DwarvenMineWorker.Action.WORK);
  }

  public void endDay() {
    makeActions(workers, DwarvenMineWorker.Action.GO_HOME, DwarvenMineWorker.Action.GO_TO_SLEEP);
  }

  private static void makeActions(Collection<DwarvenMineWorker> workers,
      DwarvenMineWorker.Action... actions) {
    for (DwarvenMineWorker worker : workers) {
      worker.action(actions);
    }
  }
}
```

### **Flyweight享元模式**

- 使用缓存来加速大量小对象的访问时间。
- 例子
    - java.lang.Integer#valueOf(int)
    - java.lang.Boolean#valueOf(boolean)

```java
public class PotionFactory {

  private final Map<PotionType, Potion> potions;

  public PotionFactory() {
    potions = new EnumMap<>(PotionType.class);
  }

  Potion createPotion(PotionType type) {
    Potion potion = potions.get(type);
    if (potion == null) {
      switch (type) {
        case HEALING:
          potion = new HealingPotion();
          potions.put(type, potion);
          break;
        case HOLY_WATER:
          potion = new HolyWaterPotion();
          potions.put(type, potion);
          break;
        case INVISIBILITY:
          potion = new InvisibilityPotion();
          potions.put(type, potion);
          break;
        default:
          break;
      }
    }
    return potion;
  }
}
```

### **Proxy代理模式**

- Provide a surrogate or placeholder for another object to control access to it.
- 例子
    - java.lang.reflect.Proxy
    - RMI

```java
public interface WizardTower {
  void enter(Wizard wizard);
}
public
 class WizardTowerProxy implements WizardTower {
  private static final int NUM_WIZARDS_ALLOWED = 3;
  private int numWizards;
  private final WizardTower tower;
  public WizardTowerProxy(WizardTower tower) {
    this.tower = tower;
  }

  @Override
  public void enter(Wizard wizard) {
    if (numWizards < NUM_WIZARDS_ALLOWED) {
      tower.enter(wizard);
      numWizards++;
    } else {
      LOGGER.info("{} is not allowed to enter!", wizard);
    }
  }
}
```

### **Abstract Factory抽象工厂模式**

- 提供一个接口创建一系列相关的对象,而不用指定具体的类型
- 例子
    - javax.xml.parsers.DocumentBuilderFactory
    - javax.xml.transform.TransformerFactory

```java
public interface Castle {
  String getDescription();
}
public interface King {
  String getDescription();
}
public interface KingdomFactory {
  Castle createCastle();
  King createKing();
}
```

### **Factory Method工厂方法模式**

- 定义一个接口用来创建对象,但是让子类决定具体要创建什么对象。
- 例子
    - java.util.Calendar
    - java.util.ResourceBundle

```java
public interface Blacksmith {
  Weapon manufactureWeapon(WeaponType weaponType);
}

public class ElfBlacksmith implements Blacksmith {
  public Weapon manufactureWeapon(WeaponType weaponType) {
    return new ElfWeapon(weaponType);
  }
}

public class OrcBlacksmith implements Blacksmith {
  public Weapon manufactureWeapon(WeaponType weaponType) {
    return new OrcWeapon(weaponType);
  }
}
```

### **Prototype原型模式**

- 基于已经存在的对象创建一个拷贝。
- 实现：implements Cloneable接口 in Java

### **Singleton单例模式**

- 确保类只有一个实例
- 实现
    - A single-element enum type is the best way to implement a singleton

```java
public enum EnumIvoryTower {
  INSTANCE;
}

public final class IvoryTower { 
    private IvoryTower() {} 
    private static final IvoryTower INSTANCE = new IvoryTower(); 
    public static IvoryTower getInstance() { 
        return INSTANCE; 
    } 
}
```

### **Command命令模式**

- 将操作封装在对象里,因此得以实现操作的参数化,队列,undo
- 实现
    
    ![graph1]({{ site.baseurl }}/assets/images/languages/design_pattern.png)
    
- [https://github.com/Cecil-Zhang/java-design-patterns/tree/master/command](https://github.com/Cecil-Zhang/java-design-patterns/tree/master/command)

- 例子
    - java.lang.Runnable
    - Netflix Hystrix

### **Observer观察者模式**

- 使一个对象可以灵活地将消息发送给感兴趣的对象
- 例子
    - java.util.Observer
    - java.util.EventListener

```java
public class Weather {
    private WeatherType currentWeather; 
    private List<WeatherObserver> observers; 
    public Weather() { 
        observers = new ArrayList<>(); 
        currentWeather = WeatherType.SUNNY; 
    } 
    public void addObserver(WeatherObserver obs) { 
        observers.add(obs); 
    } 
    public void removeObserver(WeatherObserver obs) { 
        observers.remove(obs); 
    } 
    private void notifyObservers() { 
        for (WeatherObserver obs : observers) { 
            obs.update(currentWeather); 
        } 
    } 
} 

public interface WeatherObserver { 
    void update(WeatherType currentWeather);
}
```

**Strategy策略模式**

- 使用这个模式来将一组算法封装成一系列对象。通过传递这些对象可以灵活的改变程序的功能。
    
    ![graph1]({{ site.baseurl }}/assets/images/languages/design_pattern1.png)
    
