?($	??????b?^????St$????!?QI??&??$	?.??g?@m??Z?@YuB?@!l+?M??0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$!?lV}??1?*?Թ?A8gDio??Y2??%䃮?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?QI??&??@?߾???A?d?`TR??Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-????d?]K???A A?c?]??Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????H??Zd;?O???A?Pk?w???YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????????A)?Ǻ???Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&؁sF?????m4??@??A?\?C????Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????<,??'1?Z??A?ŏ1w??Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???(\???H?z?G??A=?U?????Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?f??j+?????????A?2ı.n??Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	???<,????H?}??A??~j?t??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?	?c???T???N??Ap_?Q??Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????????????A?A?f???Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~???HP?s??A??y???Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio??+??ݓ???A??b?=??Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&e?`TR'??ё\?C???A??6?[??Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-?????鷯????A??D????Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?m4??@??V-????AΈ?????Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?8??m?????Q???A0*??D??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?"??~j??Gx$(??A.?!??u??Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE???@a??+??AtF??_??Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?St$??????|гY??A ?o_???YV-???*	     ??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????(??!      A@)333333??1M?*g??@:Preprocessing2F
Iterator::Model?Pk?w???!.?jL??@)???Mb??1c"=P9?2@:Preprocessing2U
Iterator::Model::ParallelMapV2?0?*??!??[?՘(@)?0?*??1??[?՘(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???????!?@??>Q@)H?}8g??1????'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??ܵ???!ï?Dz?0@)q?-???1?Wc"=P#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceEGr????!IT?n?@)EGr????1IT?n?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???????!rv?7@)??_vO??1????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*K?=?U??!??¯?D@)K?=?U??1??¯?D@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9J?x?~?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???H?????]?????|гY??!@?߾???	!       "	!       *	!       2$	?S?~???7С??+?? ?o_???!??6?[??:	!       B	!       J$	HP?sג????Tz8???!??u???!2??%䃮?R	!       Z$	HP?sג????Tz8???!??u???!2??%䃮?JCPU_ONLYYJ?x?~?@b Y      Y@q?O?baC@"?
both?Your program is POTENTIALLY input-bound because 51.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?38.7608% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 