$	/9?????+c???9EGr???!EGr????$	N??yu?@??25?@Զ??g?@!????x2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?g??s???s??A϶?A2??%????Y??\m????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?Zd??j?q?????A??W?2???Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?z?G????t?V??At??????Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&,Ԛ????,Ԛ????A??7??d??Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Nё\?C??jM????AгY?????Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?-?????ŏ1w-!??A?3??7???Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~?:p???=
ףp=??A\ A?c???Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?|a2U????&???A>yX?5???Y??W?2ġ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&EGr????z?):????A?W?2ı??Y??_?L??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?|?5^???Nё\?C??Ag??j+???Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??d?`T??~??k	???A?1??%???Y?ׁsF???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???S???:#J{?/??Ae?X???Y?X?? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u????=?U???AM?O???Y)\???(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&H?z?G??X9??v??A??q????Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x??#??????d?`T??Ax$(~???Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&w??/???j?t???A䃞ͪ???Y???B?i??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&c?ZB>????ǘ?????A2??%????Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ڬ?\m????QI??&??A??e?c]??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??V?/???q???h??AHP?s???Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& ?~?:p??!?rh????AP??n???Yh??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9EGr?????(\?µ?A?:pΈ??YHP?sע?*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?[ A?c??!?i??ΜA@)???Q???1???h@@:Preprocessing2F
Iterator::Modelx??#????!j??ERA@)??|гY??1?ʶ15@:Preprocessing2U
Iterator::Model::ParallelMapV29??v????!Qy??*@)9??v????1Qy??*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?-????!???VP@)h??|?5??1.?t???%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?0?*??!_3?.w@)?0?*??1_3?.w@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?>W[????!??h?>?-@)H?}8g??1x?:Fxw@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??v????!Oݽ8?93@)9??m4???1i?%n[?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*}гY????!1??6?L@)}гY????11??6?L@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?E׺?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	7?iv5O??!??so????(\?µ?!q???h??	!       "	!       *	!       2$	????k??1?y֢???:pΈ??!?W?2ı??:	!       B	!       J$	??????L2?7??$????ۗ?!??\m????R	!       Z$	??????L2?7??$????ۗ?!??\m????JCPU_ONLYY?E׺?@b 