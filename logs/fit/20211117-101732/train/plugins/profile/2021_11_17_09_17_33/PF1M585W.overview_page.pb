?($	+?'h????m???????/L?
F??!]?C?????$	??W?t?@8????	@??
`??@!???-?j0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$#J{?/L??M?O????A]m???{??Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&U0*??????ܵ??A??ͪ????Y?p=
ף??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&/?$????	h"lx??A?lV}???YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?HP?????_vO??AP?s???Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?uq????:pΈ??Am???{???Y??6???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F??f?c]?F??A?QI??&??Y??j+????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{?/L?
??]?C?????A??MbX??Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X9??v??????AE???JY??Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h"lxz????MbX9??A?sF????Y?|a2U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&		?^)???Zd;?O???A???V?/??YbX9?Ȧ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
x??#????)??0???A?k	??g??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?C??????u?????AL?
F%u??Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?):????C?i?q???A?C?l????Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t???Qk?w????A?B?i?q??Y?Zd;??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??	h??KY?8????A?L?J???Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&k+??ݓ??pΈ?????AF????x??Y?:pΈҞ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&4??7???????????A???QI???Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????pΈ?????AH?}8g??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??a??4??E???JY??A???S???Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@a??+??NbX9???AF????x??Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/L?
F??r?鷯??A鷯???Y???B?i??*	133337?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatd]?Fx??!]?0??@@)?
F%u??1YX?׎*?@:Preprocessing2F
Iterator::Model?@??ǘ??!?? !?jD@)@?߾???1?Fؤ??7@:Preprocessing2U
Iterator::Model::ParallelMapV2/?$????!?@)??0@)/?$????1?@)??0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?0?*??!<???M@)U0*?д?1?1?'6"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??	h"??!B??
X(@))\???(??1?='??]@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?Q???!?]VR@)?Q???1?]VR@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF%u???!XJډx0@)?j+??ݣ?1??3f0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?+e?X??!??D?3@)?+e?X??1??D?3@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9g?L=:?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	B?z????iZ?_??r?鷯??!?u?????	!       "	!       *	!       2$	??Z%A???>wky5??鷯???!??ͪ????:	!       B	!       J$	?R???????^???{???6???!???QI??R	!       Z$	?R???????^???{???6???!???QI??JCPU_ONLYYg?L=:?@b Y      Y@q-?}P@"?
both?Your program is POTENTIALLY input-bound because 57.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?65.9669% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 